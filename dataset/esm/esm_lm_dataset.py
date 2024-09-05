import json
import random
import copy
import torch
from transformers import EsmTokenizer

from ..data_interface import register_dataset
from ..lmdb_dataset import LMDBDataset
from ..utils import pad_sequences


@register_dataset
class EsmLMDataset(LMDBDataset):
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
	
	def __len__(self):
		return int(self._get("length"))
	
	def __getitem__(self, index:int):
		index = f"{index:09d}"
		entry = json.loads(self._get(index))
		seq = entry['seq'][:self.max_length]
		# mask sequence for training
		ids = self.tokenizer.encode(seq, add_special_tokens=False)
		tokens = self.tokenizer.convert_ids_to_tokens(ids)
		masked_tokens, labels = self._apply_bert_mask(tokens)
		masked_seq = " ".join(masked_tokens)
		return masked_seq, labels
	
	def _apply_bert_mask(self, tokens):
		masked_tokens = copy.copy(tokens)
		labels = torch.full((len(tokens)+2,), -1, dtype=torch.long) # add labels for cls and eos tokens 
		for i in range(len(tokens)):
			token = tokens[i]
			
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