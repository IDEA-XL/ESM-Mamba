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

import sys
sys.path.append('./utils')
from utils import multimodal_dataset


_10TB = 10995116277760


def train(ckpt=None):

    train_lmdb_path = "/cto_labs/liuzijing/lmdb/train_dedup/data.lmdb"
    valid_lmdb_path = "/cto_labs/liuzijing/lmdb/valid/data.lmdb"

    # train_lmdb_path = "/cto_studio/xtalpi_lab/temp/lmdb/train_dedup/data.lmdb"
    # valid_lmdb_path = "/cto_studio/xtalpi_lab/temp/lmdb/valid/data.lmdb"

    train_dataset = multimodal_dataset.SeqDataset(train_lmdb_path, 
                                                        max_length=1024)
    
    test_dataset = multimodal_dataset.SeqDataset(valid_lmdb_path, 
                                                        max_length=1024)

    batch_size = 128
    gradient_accumulation = 1
    output_dir = "/cto_labs/liuzijing/outputs/gpt_seq"
    
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

    gradient_checkpointing = True
    save_steps = 5000
    eval_steps = 5000
    save_total_limit=3

    args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        warmup_steps=5000,
        num_train_epochs=10,
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
        dataloader_num_workers=0,
        gradient_checkpointing=gradient_checkpointing,
        data_seed=54
    )


    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=multimodal_dataset.collate_fn_gpt
    )

    if ckpt is None:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=ckpt)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        ckpt = None
    elif len(sys.argv) == 2:
        ckpt = sys.argv[1]
    train(ckpt)
