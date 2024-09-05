import json
import random
import copy
import torch
import pickle
import pytorch_lightning as pl

from transformers import EsmTokenizer
from torch.utils.data import DataLoader

from ..data_interface import register_dataset
from ..lmdb_dataset import LMDBDataset
from ..utils import pad_sequences


# read fasta
def read_fasta(fasta_file):
    sequence_labels, sequence_strs = [], []
    cur_seq_label = None
    buf = []
    
    def _flush_current_seq():
        nonlocal cur_seq_label, buf
        if cur_seq_label is None:
            return
        sequence_labels.append(cur_seq_label)
        sequence_strs.append("".join(buf))
        cur_seq_label = None
        buf = []

    with open(fasta_file, "r") as infile:
        for line_idx, line in enumerate(infile):
            if line.startswith(">"):  # label line
                _flush_current_seq()
                line = line[1:].strip()
                if len(line) > 0:
                    cur_seq_label = line
                else:
                    cur_seq_label = f"seqnum{line_idx:09d}"
            else:  # sequence line
                buf.append(line.strip())
    
    _flush_current_seq()
    return sequence_labels, sequence_strs


@register_dataset
class EsmPairDataset(LMDBDataset):
	"""
	Dataset of Mask Token Reconstruction with Structure information
	"""

	def __init__(self,
	             tokenizer: str,
	             max_length: int = 512,
				 mask_ratio: float = 0.15,
				 **kwargs):
		"""

		Args:
			tokenizer: EsmTokenizer config path
			max_length: max length of sequence
			use_bias_feature: whether to use structure information
			mask_ratio: ratio of masked tokens
			**kwargs: other arguments for LMDBDataset
		"""
		super().__init__(**kwargs)
		self.tokenizer: EsmTokenizer = EsmTokenizer.from_pretrained(tokenizer)
		self.aa = [k for k in self.tokenizer.get_vocab().keys()]

		self.max_length = max_length
		self.mask_ratio = mask_ratio
		self.fasta_path = "/cto_studio/xtalpi_lab/temp/9606.protein.sequences.v12.0.fa"
		self.seq_labels, self.seq_strs = read_fasta(self.fasta_path)
		self.data_list = []
	
	def __len__(self):
		return len(self.data_list)
	
	def _load_pairs(self, path):
		with open(path, "rb") as f:
			self.data_list = pickle.load(f)

	def _dataloader(self, stage):
		self.dataloader_kwargs["shuffle"] = True if stage == "train" else False
		lmdb_path = getattr(self, f"{stage}_lmdb")
		dataset = copy.copy(self)
		dataset._load_pairs(lmdb_path)
		setattr(dataset, "stage", stage)

		return DataLoader(dataset, collate_fn=dataset.collate_fn, **self.dataloader_kwargs)

	def __getitem__(self, index:int):
		seq_label1, seq_label2 = self.data_list[index]
		seq1 = self.seq_strs[self.seq_labels.index(seq_label1)]
		seq2 = self.seq_strs[self.seq_labels.index(seq_label2)]

		seq1 = seq1[:self.max_length]
		seq2 = seq2[:self.max_length]

		# mask sequence for training
		ids1 = self.tokenizer.encode(seq1, add_special_tokens=False)
		ids2 = self.tokenizer.encode(seq2, add_special_tokens=False)
		tokens1 = self.tokenizer.convert_ids_to_tokens(ids1)
		tokens2 = self.tokenizer.convert_ids_to_tokens(ids2)
		masked_tokens, labels = self._apply_bert_mask(tokens1, tokens2)
		masked_seq = " ".join(masked_tokens)
		return masked_seq, labels
	
	def _apply_bert_mask(self, tokens1, tokens2):
		masked_tokens = copy.copy(tokens1 + [self.tokenizer.eos_token] + tokens2)

		labels = torch.full((len(masked_tokens)+2,), -1, dtype=torch.long) # add labels for cls and eos tokens
		prob_seq = random.random()
		if prob_seq < 0.5:
			# mask seq1
			start = 0
			end = len(tokens1)
		else:
			# mask seq2
			start = len(tokens1) + 1
			end = len(masked_tokens)

		for i in range(start, end):
			token = masked_tokens[i]
			
			prob = random.random()
			if prob < self.mask_ratio:
				prob /= self.mask_ratio
				labels[i+1] = self.tokenizer.convert_tokens_to_ids(token)
				
				if prob < 0.8:
					# 80% random change to mask token
					token = self.tokenizer.mask_token
					
				elif prob < 0.9:
					# 10% chance to change to random token
					token = random.choice(self.aa)
				else:
					# 10% chance to keep current token
					pass
				
				masked_tokens[i] = token
		
		return masked_tokens, labels
	
	def collate_fn(self, batch):
		seqs, label_ids = tuple(zip(*batch))

		label_ids = pad_sequences(label_ids, -1)
		labels = {"labels": label_ids}
		
		encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
		inputs = {"inputs": encoder_info}

		return inputs, labels