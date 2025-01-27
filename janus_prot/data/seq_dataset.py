import lmdb
import os
import json
import random
from typing import Union

import torch
from torch.utils.data import Dataset

from esm.tokenization import EsmSequenceTokenizer
from esm.utils import encoding
from janus_prot.data.constants import _10TB

class SeqDataset(Dataset):
    """
    Parameters:
        lmdb_path (`str`):
            Path to the sequence data.

        max_length (`int`)
    """
    def __init__(self,
                 lmdb_path: str,
	             max_length: int = 512,
                 sequence_tokenizer = EsmSequenceTokenizer()):
        super().__init__()

        self.lmdb_path = lmdb_path
        self.sequence_tokenizer = sequence_tokenizer
        # self.aa = [k for k in self.sequence_tokenizer.get_vocab().keys()]
        self.max_length = max_length

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
        return self.len_seq
    
    def __getitem__(self, index:int):
        index = random.randint(0, self.len_seq-1)
        index = f"{index:09d}"
        entry = json.loads(self._get(index))
        seq = entry['seq'][:self.max_length]
        token_ids = encoding.tokenize_sequence(
                seq, self.sequence_tokenizer, add_special_tokens=True
            ) # <bos> AA <eos>
        labels = torch.full((len(token_ids),), -100, dtype=torch.long)
        for i in range(len(token_ids)):
            labels[i] = token_ids[i]
        return token_ids, labels
    

class ProgenSeqDataset(Dataset):
    """
    Sequqnce dataset for progen in the format of "1ABC2" or "2CBA1"
    Parameters:
        lmdb_path (`str`):
            Path to the sequence data.

        max_length (`int`)
    """
    def __init__(self,
                 lmdb_path: str,
	             max_length: int = 512,
                 sequence_tokenizer = EsmSequenceTokenizer()):
        super().__init__()

        self.lmdb_path = lmdb_path
        self.sequence_tokenizer = sequence_tokenizer
        # self.aa = [k for k in self.sequence_tokenizer.get_vocab().keys()]
        self.max_length = max_length

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
        return self.len_seq
    
    def __getitem__(self, index:int):
        index = random.randint(0, self.len_seq-1)
        index = f"{index:09d}"
        entry = json.loads(self._get(index))
        if len(entry['seq']) > self.max_length - 2:
            start_offset = random.randint(0, max(0, len(entry['seq']) - self.max_length+2))
            seq = entry['seq'][start_offset:start_offset + self.max_length-2]
        else:
            seq = entry['seq'][:self.max_length-2]
        if random.random() > 0.5:
            seq = "1" + seq + "2"
        else:
            seq = "2" + seq[::-1] + "1"

        token_ids = torch.tensor(self.sequence_tokenizer.encode(seq).ids, dtype=torch.long)
        labels = torch.full((len(token_ids),), -100, dtype=torch.long)
        for i in range(len(token_ids)):
            labels[i] = token_ids[i]
        return token_ids, labels