import random
import lmdb
import os
import json
import pickle
import glob
import pandas as pd
from typing import Union

import torch
from torch.utils.data import Dataset

from esm.tokenization import EsmSequenceTokenizer, StructureTokenizer
from esm.utils import encoding
from janus_prot.data.constants import _10TB

class Seq40StructDataset(Dataset):
    """
    sequence to structure dataset in the format of "1ABC23sAsB4" or "2CBA14sBsA3"
    sampled by sequence identity 40%
    Parameters:
        lmdb_path (`str`):
            Path to the sequence data.

        struct_path (`str`):
            Path to the structure token data.

        max_length (`int`)

        struct_only (`bool`):
            Whether to use the sequence GPT loss.
    """
    def __init__(self,
                 lmdb_path: str,
	             struct_path: list,
	             max_length: int = 512,
                 seq_ratio: int = 1,
				 struct_only: bool = False,
                 sequence_tokenizer = EsmSequenceTokenizer()):
        super().__init__()

        self.seq_ratio = seq_ratio

        self.lmdb_path = lmdb_path
        self.sequence_tokenizer = sequence_tokenizer
        self.structure_tokenizer = StructureTokenizer()
        self.struct_inf_token_id = 2246
        self.max_length = max_length
        self.struct_only = struct_only

        self.struct_token = {}
        self.struct_seq = {}
        self.cluster2id = {}
        self.clusters = {}
        data_names = ["AF2_ebi", "PDB"]
        self.data_names = data_names
        self.len_struct = {}
        for folder in struct_path:
            if data_names[0] in folder:
                data_idx = 0
                self.cluster2id[data_names[data_idx]] = {}
                cluster_path = os.path.join(folder, "clusterRes_cluster.tsv")
                fasta_path = os.path.join(folder, "af2_ebi.fasta")
                struct_path = os.path.join(folder, "af2_ebi_str.pkl")
            elif data_names[1] in folder:
                data_idx = 1
                self.cluster2id[data_names[data_idx]] = {}
                cluster_path = os.path.join(folder, "clusterRes_cluster.tsv")
                fasta_path = os.path.join(folder, "pdb20220928.fasta")
                struct_path = os.path.join(folder, "pdb20220928_str1.pkl")

            df_cluster = pd.read_csv(cluster_path, sep="\t", header=None)
            for x in df_cluster.itertuples():
                cluster, member = x[1], x[2]
                if cluster not in self.cluster2id[data_names[data_idx]]:
                    self.cluster2id[data_names[data_idx]][cluster] = [member]
                else:
                    self.cluster2id[data_names[data_idx]][cluster].append(member)
            self.len_struct[data_names[data_idx]] = len(self.cluster2id[data_names[data_idx]])
            self.clusters[data_names[data_idx]] = list(self.cluster2id[data_names[data_idx]].keys())

            with open(struct_path, 'rb') as f:
                self.struct_token[data_names[data_idx]] = pickle.load(f)

            uniprot = []
            fasta_seqs = []
            with open(fasta_path, "r") as f:
                for line in f:
                    if line.startswith('>'):
                        uniprot.append(line.strip())
                    else:
                        fasta_seqs.append(line.strip())
                self.struct_seq[data_names[data_idx]] = dict(zip(uniprot, fasta_seqs))

        self.env = None
        self.txn = None

        self._init_lmdb(lmdb_path)
        self.len_seq = int(self._get("length"))

    def _init_lmdb(self, path):
        if self.env is not None:
            self._close_lmdb()

        # open lmdb
        self.env = lmdb.open(
            path, subdir=os.path.isdir(path), lock=False, readonly=False,
            readahead=False, meminit=False, map_size=_10TB, max_readers=1,
        )
        self.txn = self.env.begin(write=False, buffers=True)
    
    def _close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.txn = None

    def _cursor(self):
        return self.operator.cursor()

    def _get(self, key: Union[str, int]):
        value = self.txn.get(str(key).encode())
        
        if value is not None:
            value = value.tobytes()
        
        return value

    def __len__(self):
        """
        the length is the length of smaller dataset (PDB dataset)
        """
        return int(min(list(self.len_struct.values())))
    
    def __getitem__(self, index:int):
        if index % self.seq_ratio == 0: # 1/seq_ratio prob. using structure
            index = index // self.seq_ratio
            data_idx = random.randint(0,1)
            data_key = self.data_names[data_idx]

            cluster_idx = random.randint(0, self.len_struct[data_key] - 1)
            clust = self.clusters[data_key][cluster_idx]
            prot_ids = self.cluster2id[data_key][clust]
            seqs = []
            for pid in prot_ids:
                seqs.append(self.struct_seq[data_key][">"+pid])
            seqs_set = list(set(seqs))
            tmp_idx = random.randint(0, len(seqs_set) - 1)
            seq = seqs_set[tmp_idx]
            prot_id = prot_ids[seqs.index(seq)]
            struct = self.struct_token[data_key][prot_id]
            
            # cut to max_len/2 - 2 = 510  <bos> AAA <eos> <S_BOS> SSS <S_EOS>
            start_offset = random.randint(0, max(0, len(seq) - self.max_length//2+2))
            start_idx = start_offset
            end_idx = start_idx + min(self.max_length//2-2, len(seq))

            struct_seq = torch.tensor(struct[start_idx:end_idx], dtype=torch.int64)
            seq = seq[start_idx:end_idx]

            # seq+structure 1 AB 2 3 SaSb 4 or 2 BA 1 4 SbSa 3
            if random.random() > 0.5:
                seq = "1" + seq + "2" + "3" + seq + "4"
            else:
                seq = "2" + seq[::-1] + "1" + "4" + seq + "3"
                struct_seq = torch.flip(struct_seq, dims=[0])

            token_ids = torch.tensor(self.sequence_tokenizer.encode(seq).ids, dtype=torch.long)
            structure_seq_mask: torch.BoolTensor = token_ids == -100
            structure_seq_mask[len(struct_seq)+3:-1] = True
            
            labels = torch.full((len(token_ids),), -100, dtype=torch.long)
            labels[structure_seq_mask] = struct_seq

            struct_seq[struct_seq == -100] = 2246
            token_ids[structure_seq_mask] = struct_seq

            return token_ids, labels, structure_seq_mask
        else:  # sequence
            index = random.randint(0, self.len_seq-1)
            index = f"{index:09d}"
            entry = json.loads(self._get(index))
            seq = entry['seq'][:self.max_length]
            token_ids = encoding.tokenize_sequence(
                    seq, self.sequence_tokenizer, add_special_tokens=True
                )        # <bos> AA <eos>
            labels = torch.full((len(token_ids),), -100, dtype=torch.long)
            for i in range(len(token_ids)):
                labels[i] = token_ids[i]
            return token_ids, labels   


class SeqStructMixDataset(torch.utils.data.Dataset):
    """
    Structure and sequence dataset 
    seq2struct in the format of "1ABC23sAsBsC4" or "2CBA14sCsBsA3";
    sampled by sequence identity 40%;
    struct2seq in the format of "3[mask][mask][mask]41ABC2"
    and the structure embeddings
    Parameters:
        lmdb_path (`str`):
            Path to the sequence data.

        struct_path (`list`):
            Paths to the structure token datasets.

        struct2seq_path (`str`):
            Path to the structure embedding data.

        max_length (`int`)

        seq_ratio: (`float`):
            the ratio of seq2struct and struct2seq samples
            the seq2struct data percentage is seq_ratio x 100%
    """
    def __init__(self,
                 lmdb_path: str,
	             struct_path: list,
                 struct2seq_path: str,
	             max_length: int = 512,
                 seq_ratio: float = 1.0,
                 sequence_tokenizer = EsmSequenceTokenizer()):
        super().__init__()

        self.seq_ratio = seq_ratio

        self.lmdb_path = lmdb_path
        self.sequence_tokenizer = sequence_tokenizer
        self.structure_tokenizer = StructureTokenizer()
        self.struct_inf_token_id = 2246
        self.max_length = max_length
        self.is_flip = False

        ### struct token data
        self.struct_token = {}
        self.struct_seq = {}
        self.cluster2id = {}
        self.clusters = {}
        data_names = ["AF2_ebi", "PDB"]
        self.data_names = data_names
        self.len_struct = {}
        for folder in struct_path:
            if data_names[0] in folder:
                data_idx = 0
                self.cluster2id[data_names[data_idx]] = {}
                cluster_path = os.path.join(folder, "clusterRes_cluster.tsv")
                fasta_path = os.path.join(folder, "af2_ebi.fasta")
                struct_path = os.path.join(folder, "af2_ebi_str.pkl")
            elif data_names[1] in folder:
                data_idx = 1
                self.cluster2id[data_names[data_idx]] = {}
                cluster_path = os.path.join(folder, "clusterRes_cluster.tsv")
                fasta_path = os.path.join(folder, "pdb20220928.fasta")
                struct_path = os.path.join(folder, "pdb20220928_str1.pkl")

            df_cluster = pd.read_csv(cluster_path, sep="\t", header=None)
            for x in df_cluster.itertuples():
                cluster, member = x[1], x[2]
                if cluster not in self.cluster2id[data_names[data_idx]]:
                    self.cluster2id[data_names[data_idx]][cluster] = [member]
                else:
                    self.cluster2id[data_names[data_idx]][cluster].append(member)
            self.len_struct[data_names[data_idx]] = len(self.cluster2id[data_names[data_idx]])
            self.clusters[data_names[data_idx]] = list(self.cluster2id[data_names[data_idx]].keys())

            with open(struct_path, 'rb') as f:
                self.struct_token[data_names[data_idx]] = pickle.load(f)

            uniprot = []
            fasta_seqs = []
            with open(fasta_path, "r") as f:
                for line in f:
                    if line.startswith('>'):
                        uniprot.append(line.strip())
                    else:
                        fasta_seqs.append(line.strip())
                self.struct_seq[data_names[data_idx]] = dict(zip(uniprot, fasta_seqs))

        ### struct embedding data
        # struct2seq_path = "/cto_studio/xtalpi_lab/temp/swiss_prot_esm3"
        self.name_list = []
        self.name_to_batch = {}
        self.struct2seq_path = struct2seq_path
        for k in range(8):
            f_list = glob.glob(f'{struct2seq_path}/batch_{k}' + '/AF-*_v4.pkl')
            for f_name in f_list:
                uniprot = f_name.split('/')[-1].split(".")[0]
                self.name_list.append(uniprot)
                self.name_to_batch[uniprot] = k

        self.len_emb = len(self.name_list)
       
        self.env = None
        self.txn = None

        self._init_lmdb(lmdb_path)
        self.len_seq = int(self._get("length"))

    def _init_lmdb(self, path):
        if self.env is not None:
            self._close_lmdb()

        # open lmdb
        self.env = lmdb.open(
            path, subdir=os.path.isdir(path), lock=False, readonly=False,
            readahead=False, meminit=False, map_size=_10TB, max_readers=1,
        )
        self.txn = self.env.begin(write=False, buffers=True)
    
    def _close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.txn = None

    def _cursor(self):
        return self.operator.cursor()

    def _get(self, key: Union[str, int]):
        value = self.txn.get(str(key).encode())
        
        if value is not None:
            value = value.tobytes()
        
        return value

    def __len__(self):
        """
        the length is the length of smaller dataset (PDB dataset)
        """
        return int(min(list(self.len_struct.values())))
    
    def __getitem__(self, index:int):
        prob = random.random()
        if prob < self.seq_ratio: # seq_ratio prob. seq to struct
            data_idx = random.randint(0,1)
            if prob > 0.4: # 10% de novo structure, use AF_EBI dataset
                data_idx = 0
            data_key = self.data_names[data_idx]

            cluster_idx = random.randint(0, self.len_struct[data_key] - 1)
            clust = self.clusters[data_key][cluster_idx]
            prot_ids = self.cluster2id[data_key][clust]
            seqs = []
            for pid in prot_ids:
                seqs.append(self.struct_seq[data_key][">"+pid])
            seqs_set = list(set(seqs))
            tmp_idx = random.randint(0, len(seqs_set) - 1)
            seq = seqs_set[tmp_idx]
            prot_id = prot_ids[seqs.index(seq)]
            struct = self.struct_token[data_key][prot_id]
            
            # cut to max_len/2 - 2 = 510  <bos> AAA <eos> <S_BOS> SSS <S_EOS>
            start_offset = random.randint(0, max(0, len(seq) - self.max_length//2+2))
            start_idx = start_offset
            end_idx = start_idx + min(self.max_length//2-2, len(seq))

            struct_seq = torch.tensor(struct[start_idx:end_idx], dtype=torch.int64)
            seq = seq[start_idx:end_idx]

            # seq+structure 1 AB 2 4096 SaSb 4097
            if prob < 0.4: # 40% seq to structure
                position_ids = torch.arange(len(seq)+2)
                position_ids = torch.cat((position_ids, position_ids))
                if self.is_flip and random.random() > 0.5:
                    seq = "2" + seq[::-1] + "1" + "<|s_eos|>" + seq + "<|s_bos|>"
                    struct_seq = torch.flip(struct_seq, dims=[0])
                else:
                    seq = "1" + seq + "2" + "<|s_bos|>" + seq + "<|s_eos|>"
            else:  # 10% de novo structure
                position_ids = torch.arange(len(seq)+2)
                seq = "<|s_bos|>" + seq + "<|s_eos|>"

            token_ids = torch.tensor(self.sequence_tokenizer.encode(seq).ids, dtype=torch.long)
            structure_seq_mask: torch.BoolTensor = token_ids == -100
            structure_seq_mask[-(len(struct_seq)+1):-1] = True

            token_ids[structure_seq_mask] = struct_seq 
            labels = torch.full((len(token_ids),), -100, dtype=torch.long)
            structure_seq_mask[-1] = True
            labels[structure_seq_mask] = token_ids[structure_seq_mask]
            structure_seq_mask[-1] = False
            struct_seq[struct_seq == -100] = self.struct_inf_token_id
            token_ids[structure_seq_mask] = struct_seq
            structure_seq_mask[-1] = True
            structure_seq_mask[-(len(struct_seq)+2)] = True

            return token_ids, labels, structure_seq_mask, position_ids 
        elif prob < 0.75:  # 25% struct to sequence
            index = random.randint(0, self.len_emb-1)
            key = self.name_list[index]
            k = self.name_to_batch[key]
            fname = f'{self.struct2seq_path}/batch_{k}' + f'/{key}.pkl'
            with open(fname, 'rb') as f:
                x_data = pickle.load(f)
                plddt = x_data["plddt"]
                emb = x_data["emb"]
                seq = x_data["seq"]
            
            struct_emb = token_ids = torch.tensor(emb, dtype=torch.float)
            position_ids = torch.arange(len(seq)+2)
            position_ids = torch.cat((position_ids, position_ids))
            if self.is_flip and random.random() > 0.5:
                seq = "4" + seq + "3" + "2" + seq[::-1] + "1"
                struct_emb = torch.flip(struct_emb, dims=[0])
            else:
                seq = "3" + seq + "4" + "1" + seq + "2"

            token_ids = torch.tensor(self.sequence_tokenizer.encode(seq).ids, dtype=torch.long)

            struct_emb_mask: torch.BoolTensor = token_ids == -100
            struct_emb_mask[1:len(plddt)+1] = True

            structure_seq_mask = torch.full((len(token_ids),), False, dtype=torch.bool)

            labels = torch.full((len(token_ids),), -100, dtype=torch.long)
            labels[len(plddt)+3:] = token_ids[len(plddt)+3:]

            return token_ids, labels, structure_seq_mask, position_ids, struct_emb_mask, struct_emb
        
        else: # 25% pure sequence
            index = random.randint(0, self.len_seq-1)
            index = f"{index:09d}"
            entry = json.loads(self._get(index))
            seq = entry['seq'][:self.max_length-2]
            position_ids = torch.arange(len(seq)+2)
            if self.is_flip and random.random() > 0.5:
                seq = "2" + seq[::-1] + "1"
            else:
                seq = "1" + seq + "2"
            token_ids = torch.tensor(self.sequence_tokenizer.encode(seq).ids, dtype=torch.long)
            labels = torch.full((len(token_ids),), -100, dtype=torch.long)
            for i in range(len(token_ids)):
                labels[i] = token_ids[i]
            structure_seq_mask = torch.full((len(token_ids),), False, dtype=torch.bool)
            return token_ids, labels, structure_seq_mask, position_ids, 1


class SeqStructureDataset(Dataset):
    """
    sequence to structure dataset in the format of "1ABC23sAsB4" or "2CBA14sBsA3"
    Parameters:
        lmdb_path (`str`):
            Path to the sequence data.

        struct_path (`str`):
            Path to the structure token data.

        max_length (`int`)

        is_flip (`bool`):
            Whether to randomly flip the sequence.
    """
    def __init__(self,
                 lmdb_path: str,
	             struct_path: str,
	             max_length: int = 512,
				 is_flip: bool = False,
                 sequence_tokenizer = EsmSequenceTokenizer()):
        super().__init__()

        self.lmdb_path = lmdb_path
        self.sequence_tokenizer = sequence_tokenizer
        self.structure_tokenizer = StructureTokenizer()
        # self.aa = [k for k in self.sequence_tokenizer.get_vocab().keys()]
        self.max_length = max_length
        self.is_flip = is_flip

        with open(struct_path, 'rb') as f:
            self.struct_data = pickle.load(f)
        
        self.len_struct = len(self.struct_data)
        self.struct_seq = list(self.struct_data.keys())

        self.env = None
        self.txn = None

        self._init_lmdb(lmdb_path)
        self.len_seq = int(self._get("length"))

    def _init_lmdb(self, path):
        if self.env is not None:
            self._close_lmdb()

        # open lmdb
        self.env = lmdb.open(
            path, subdir=os.path.isdir(path), lock=False, readonly=False,
            readahead=False, meminit=False, map_size=_10TB, max_readers=1,
        )
        self.txn = self.env.begin(write=False, buffers=True)
    
    def _close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.txn = None

    def _cursor(self):
        return self.operator.cursor()

    def _get(self, key: Union[str, int]):
        value = self.txn.get(str(key).encode())
        
        if value is not None:
            value = value.tobytes()
        
        return value

    def __len__(self):
        return int(self.len_struct)
    
    def __getitem__(self, index:int):
        index = index // self.seq_ratio
        seq = self.struct_seq[index]
        struct = self.struct_data[seq] # sAsBsC
        # cut to max_len/2 - 2 = 510  <bos> AAA <eos> <S_BOS> SSS <S_EOS>
        start_offset = random.randint(0, max(0, len(seq) - self.max_length//2+2))
        start_idx = start_offset
        end_idx = start_idx + min(self.max_length//2-2, len(seq))

        struct_seq = torch.tensor(struct[start_idx:end_idx], dtype=torch.int64)
        seq = seq[start_idx:end_idx]

        # seq+structure 1 AB 2 4096 SaSb 4097 
        position_ids = torch.arange(len(seq)+2)
        position_ids = torch.cat((position_ids,position_ids))
        if self.is_flip and random.random() > 0.5:
            seq = "2" + seq[::-1] + "1" + "<|s_eos|>" + seq + "<|s_bos|>"
            struct_seq = torch.flip(struct_seq, dims=[0])
        else:
            seq = "1" + seq + "2" + "<|s_bos|>" + seq + "<|s_eos|>"
            position_ids = torch.arange(len(seq)+2)
            position_ids = torch.cat((position_ids,position_ids))

        token_ids = torch.tensor(self.sequence_tokenizer.encode(seq).ids, dtype=torch.long)
        structure_seq_mask: torch.BoolTensor = token_ids == -100
        structure_seq_mask[len(struct_seq)+3:-1] = True
        token_ids[structure_seq_mask] = struct_seq

        labels = torch.full((len(token_ids),), -100, dtype=torch.long)
        structure_seq_mask[-1] = True
        labels[structure_seq_mask] = token_ids[structure_seq_mask]
        structure_seq_mask[len(struct_seq)+2] = True

        return token_ids, labels, structure_seq_mask, position_ids
        

class LimitedSeqStructureDataset(Dataset):
    """
    Selected structure and sequence dataset in the format of "1ABC23sAsB4" or "2CBA14sBsA3"
    For testing of overfitting
    Parameters:
        lmdb_path (`str`):
            Path to the sequence data.

        struct_path (`str`):
            Path to the structure token data.

        max_length (`int`)

        struct_only (`bool`):
            Whether to use the sequence GPT loss.
    """
    def __init__(self,
                 lmdb_path: str,
	             struct_path: str,
	             max_length: int = 512,
                 seq_ratio: int = 1,
				 struct_only: bool = False,
                 sequence_tokenizer = EsmSequenceTokenizer()):
        super().__init__()

        self.seq_ratio = seq_ratio

        self.lmdb_path = lmdb_path
        self.sequence_tokenizer = sequence_tokenizer
        self.structure_tokenizer = StructureTokenizer()
        # self.aa = [k for k in self.sequence_tokenizer.get_vocab().keys()]
        self.max_length = max_length
        self.struct_only = struct_only

        with open(struct_path, 'rb') as f:
            self.struct_data = pickle.load(f)
        
        self.len_struct = len(self.struct_data)
        self.struct_seq = list(self.struct_data.keys())

        self.indices = [3, 1, 90337, 2, 242712, 4, 90336, 90338]

        self.env = None
        self.txn = None

        self._init_lmdb(lmdb_path)
        self.len_seq = int(self._get("length"))

    def _init_lmdb(self, path):
        if self.env is not None:
            self._close_lmdb()

        # open lmdb
        self.env = lmdb.open(
            path, subdir=os.path.isdir(path), lock=False, readonly=False,
            readahead=False, meminit=False, map_size=_10TB, max_readers=1,
        )
        self.txn = self.env.begin(write=False, buffers=True)
    
    def _close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.txn = None

    def _cursor(self):
        return self.operator.cursor()

    def _get(self, key: Union[str, int]):
        value = self.txn.get(str(key).encode())
        
        if value is not None:
            value = value.tobytes()
        
        return value

    def __len__(self):
        return int(len(self.indices))
    
    def __getitem__(self, index:int):
        if index % self.seq_ratio == 0: # 1/seq_ratio prob. using structure
            index = index // self.seq_ratio
            idx = index % len(self.indices)
            index = self.indices[idx]
            seq = self.struct_seq[index]
            struct = self.struct_data[seq] # <S_BOS> SS <S_EOS>
            # cut to max_len/2 - 2 = 510  <bos> AAA <eos> <S_BOS> SSS <S_EOS>
            start_offset = random.randint(0, max(0, len(seq) - self.max_length//2+2))
            start_idx = start_offset
            end_idx = start_idx + min(self.max_length//2-2, len(seq))

            struct_seq = torch.tensor(struct[start_idx:end_idx], dtype=torch.int64)
            seq = seq[start_idx:end_idx]

            # seq+structure 1 AB 2 3 SaSb 4 or 2 BA 1 4 SbSa 3
            if random.random() > 0.5:
                seq = "1" + seq + "2" + "3" + seq + "4"
            else:
                seq = "2" + seq[::-1] + "1" + "4" + seq + "3"
                struct_seq = torch.flip(struct_seq, dims=[0])

            token_ids = torch.tensor(self.sequence_tokenizer.encode(seq).ids, dtype=torch.long)
            structure_seq_mask: torch.BoolTensor = token_ids == -100
            structure_seq_mask[len(struct_seq)+3:-1] = True
            token_ids[structure_seq_mask] = struct_seq

            labels = torch.full((len(token_ids),), -100, dtype=torch.long)
            labels[structure_seq_mask] = struct_seq

            return token_ids, labels, structure_seq_mask