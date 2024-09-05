import torch 
import random
import copy
import lmdb
import os
import json

import numpy as np
from typing import Union
from transformers import EsmTokenizer, EsmForMaskedLM, EsmConfig
from transformers import TrainingArguments, Trainer

from transformers import LlamaForCausalLM, LlamaConfig

_10TB = 10995116277760

class SeqInMemoryDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list: list,
	             tokenizer: str,
	             max_length: int = 512,
				 mask_ratio: float = 0.15,
				 **kwargs):
        super().__init__()
        self.data_list = data_list
        self.tokenizer: EsmTokenizer = EsmTokenizer.from_pretrained(tokenizer)
        self.aa = [k for k in self.tokenizer.get_vocab().keys()]
        self.max_length = max_length
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index:int):
        entry = self.data_list[index]
        seq = entry['seq'][:self.max_length]
        masked_seq = " ".join(seq)
        
        ids = self.tokenizer.encode(seq, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        labels = torch.full((len(tokens)+2,), -100, dtype=torch.long)
        for i in range(len(tokens)):
            token = tokens[i]
            labels[i+1] = self.tokenizer.convert_tokens_to_ids(token)
        
        return masked_seq, labels
        
        # # mask sequence for training
        # ids = self.tokenizer.encode(seq, add_special_tokens=False)
        # tokens = self.tokenizer.convert_ids_to_tokens(ids)
        # masked_tokens, labels = self._apply_bert_mask(tokens)
        # masked_seq = " ".join(masked_tokens)
        # return masked_seq, labels
    
    def _apply_bert_mask(self, tokens):
        masked_tokens = copy.copy(tokens)
        labels = torch.full((len(tokens)+2,), -100, dtype=torch.long)
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
    

class SeqLMDBDataset(torch.utils.data.Dataset):
    def __init__(self,
                 lmdb_path: str,
	             tokenizer: str,
	             max_length: int = 512,
				 mask_ratio: float = 0.15,
				 **kwargs):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.tokenizer: EsmTokenizer = EsmTokenizer.from_pretrained(tokenizer)
        self.aa = [k for k in self.tokenizer.get_vocab().keys()]
        self.max_length = max_length
        self.mask_ratio = mask_ratio
        self.env = None
        self.txn = None

        self._init_lmdb(lmdb_path)

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
        return int(self._get("length"))
    
    def __getitem__(self, index:int):
        index = f"{index:09d}"
        entry = json.loads(self._get(index))
        seq = entry['seq'][:self.max_length]
        # mask sequence for training
        ids = self.tokenizer.encode(seq, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        
        masked_seq = " ".join(seq)
        labels = torch.full((len(tokens)+2,), -100, dtype=torch.long)
        for i in range(len(tokens)):
            token = tokens[i]
            labels[i+1] = self.tokenizer.convert_tokens_to_ids(token)
        
        return masked_seq, labels
        
        # masked_tokens, labels = self._apply_bert_mask(tokens)
        # masked_seq = " ".join(masked_tokens)
        # return masked_seq, labels
    
    def _apply_bert_mask(self, tokens):
        masked_tokens = copy.copy(tokens)
        labels = torch.full((len(tokens)+2,), -100, dtype=torch.long)
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


def pad_sequences(sequences, constant_value=0, dtype=None) -> np.ndarray:
	batch_size = len(sequences)
	shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

	if dtype is None:
		dtype = sequences[0].dtype

	if isinstance(sequences[0], np.ndarray):
		array = np.full(shape, constant_value, dtype=dtype)
	elif isinstance(sequences[0], torch.Tensor):
		device = sequences[0].device
		array = torch.full(shape, constant_value, dtype=dtype, device=device)

	for arr, seq in zip(array, sequences):
		arrslice = tuple(slice(dim) for dim in seq.shape)
		arr[arrslice] = seq

	return array


def train():
    fasta_path = "/cto_studio/xtalpi_lab/benchmark/docking/data_hl/hl_test90"
    data_list = []
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith(">"):
                tmp = {}
                tmp["description"] = line[1:].strip()
            else:
                tmp["seq"] = line.strip()
                data_list.append(tmp)
    
    model_checkpoint = "/cto_labs/AIDD/WEIGHTS/Protein/esm2_t6_8M_UR50D"
    model_checkpoint = "/cto_studio/xtalpi_lab/liuzijing/weights/esm2_t30_150M_UR50D"

    config = EsmConfig.from_pretrained(model_checkpoint)

    tokenizer = EsmTokenizer.from_pretrained(model_checkpoint)
    # model = EsmForMaskedLM.from_pretrained(model_checkpoint)
    # model = EsmForMaskedLM(config)
    
    
    configuration = LlamaConfig()
    ## 150M
    configuration.hidden_size = 640
    configuration.intermediate_size = 2560
    configuration.max_position_embeddings = 1026
    configuration.num_attention_heads = 20
    configuration.num_hidden_layers = 30
    configuration.num_key_value_heads = 20
    configuration.vocab_size = 33
    configuration.bos_token_id = 0
    
    configuration.use_cache = False
    
    model = LlamaForCausalLM(configuration)

    batch_size = 16
    ddp = True
    gradient_checkpointing = True
    save_steps = 5000
    eval_steps = 5000
    save_total_limit=3
    output_dir = "./results"


    args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        warmup_steps=3000,
        num_train_epochs=3,
        # max_steps=500000,
        learning_rate=4e-4,
        fp16=True,
        logging_steps=1000,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        # ddp_find_unused_parameters=True,
        report_to="tensorboard",
        run_name=None,
        dataloader_num_workers=8,
        gradient_checkpointing=gradient_checkpointing
    )

    def collate_fn(batch):
        seqs, label_ids = tuple(zip(*batch))

        label_ids = pad_sequences(label_ids, -100)
        # labels = {"labels": label_ids}
        
        encoder_info = tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
        inputs = encoder_info
        inputs["labels"] = label_ids
        return inputs
    
    def collate_fn_gpt(batch):
        seqs, label_ids = tuple(zip(*batch))

        label_ids = pad_sequences(label_ids, -100)
        # labels = {"labels": label_ids}
        
        encoder_info = tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
        inputs = encoder_info
        inputs["labels"] = label_ids
        return inputs
    
    # train_dataset = SeqInMemoryDataset(data_list[0:256], tokenizer=model_checkpoint, max_length=1024)
    # test_dataset = SeqInMemoryDataset(data_list[256:], tokenizer=model_checkpoint, max_length=1024)
    
    # breakpoint()
    
    # train_lmdb_path = "/cto_labs/liuzijing/lmdb/train_dedup/data.lmdb"
    # valid_lmdb_path = "/cto_labs/liuzijing/lmdb/valid/data.lmdb"
    
    train_lmdb_path = "/cto_studio/xtalpi_lab/temp/lmdb/train_dedup/data.lmdb"
    valid_lmdb_path = "/cto_studio/xtalpi_lab/temp/lmdb/valid/data.lmdb"

    train_dataset = SeqLMDBDataset(train_lmdb_path, tokenizer=model_checkpoint, max_length=1024)
    test_dataset = SeqLMDBDataset(valid_lmdb_path, tokenizer=model_checkpoint, max_length=1024)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn_gpt
    )

    trainer.train()


if __name__ == "__main__":
    train()
