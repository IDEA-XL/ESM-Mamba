import torch 
import random
import copy
import lmdb
import os
import json
import pickle

import numpy as np
from typing import Union

from esm.tokenization import EsmSequenceTokenizer, StructureTokenizer
from esm.utils import encoding

_10TB = 10995116277760
SEQ_OFFSET = 33

class SeqStructDataset(torch.utils.data.Dataset):
    def __init__(self,
                 lmdb_path: str,
	             struct_path: str,
	             max_length: int = 512,
				 **kwargs):
        super().__init__()

        self.seq_ratio = 1

        self.lmdb_path = lmdb_path
        self.sequence_tokenizer = EsmSequenceTokenizer()
        self.structure_tokenizer = StructureTokenizer()
        self.aa = [k for k in self.sequence_tokenizer.get_vocab().keys()]
        self.max_length = max_length

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
        return int(self.seq_ratio * self.len_struct)
    
    def __getitem__(self, index:int):
        if index % self.seq_ratio == 0: # 1/seq_ratio prob. using structure
            index = index // self.seq_ratio
            seq = self.struct_seq[index]
            struct = self.struct_data[seq] # <S_BOS> SS <S_EOS>
            # current cut to max len of 512  # SS <S_EOS>
            start_idx = random.randint(0, max(0, len(seq) - self.max_length//2))
            end_idx = start_idx + min(self.max_length//2, len(seq))

            struct = torch.tensor(np.concatenate((struct[start_idx+1:end_idx+1], struct[-1:])), dtype=torch.int64)
            seq = seq[start_idx:end_idx]
            sequence_tokens = encoding.tokenize_sequence(seq, self.sequence_tokenizer, add_special_tokens=True)        # <bos> AA <eos>
            
            # <bos> AA <eos> SS <S_EOS> 
            token_ids = torch.cat((sequence_tokens, struct+SEQ_OFFSET))
            labels = torch.full((len(token_ids),), -100, dtype=torch.long)
            for i in range(len(sequence_tokens)-2):
                labels[i+1] = token_ids[i+1]
                labels[i+end_idx-start_idx+2] = token_ids[i+end_idx-start_idx+2]
            # TODO random cut to max len of 512 

        else:  # sequence
            index = random.randint(0, self.len_seq-1)
            index = f"{index:09d}"
            entry = json.loads(self._get(index))
            seq = entry['seq'][:self.max_length]
            token_ids = encoding.tokenize_sequence(
                    seq, self.sequence_tokenizer, add_special_tokens=True
                )        # <bos> AA <eos>
            labels = torch.full((len(token_ids),), -100, dtype=torch.long)
            for i in range(len(token_ids)-2):
                labels[i+1] = token_ids[i+1]
        
        return token_ids, labels
        

def pad_sequences(sequences, constant_value=0, dtype=None, return_mask=False) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
        attention_mask = np.full(shape, 0, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        device = sequences[0].device
        array = torch.full(shape, constant_value, dtype=dtype, device=device)
        attention_mask = torch.full(shape, 0, dtype=dtype, device=device)

    for mask, arr, seq in zip(attention_mask, array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq
        mask[arrslice] = 1

    if return_mask:
        return array, attention_mask
    else:
        return array


def collate_fn_gpt(batch):
    inputs = {}
    seqs, label_ids = tuple(zip(*batch))

    label_ids = pad_sequences(label_ids, -100)
    # labels = {"labels": label_ids}
    pad_token_id = EsmSequenceTokenizer().pad_token_id
    
    seqs, masks = pad_sequences(seqs, pad_token_id, return_mask=True)
    inputs["input_ids"] = seqs
    inputs["labels"] = label_ids
    inputs["attention_mask"] = masks
    return inputs
    