import torch 
import random
import copy
import lmdb
import os
import json

import numpy as np
from torch import nn
from typing import Union
from tokenizers import Tokenizer

from transformers import EsmTokenizer, EsmForMaskedLM, EsmConfig
from transformers import TrainingArguments, Trainer

from transformers import LlamaForCausalLM, LlamaConfig

from model.progen.modeling_progen import ProGenForCausalLM
from model.progen.configuration_progen import ProGenConfig

import sys
sys.path.append('./utils')
from utils import multimodal_dataset


_10TB = 10995116277760


def train(ckpt=None):

    # train_lmdb_path = "/cto_labs/liuzijing/lmdb/train_dedup/data.lmdb"
    # valid_lmdb_path = "/cto_labs/liuzijing/lmdb/valid/data.lmdb"
    # output_dir = "/cto_labs/liuzijing/outputs/progen_seq"

    train_lmdb_path = "/cto_studio/xtalpi_lab/temp/lmdb/train_dedup/data.lmdb"
    valid_lmdb_path = "/cto_studio/xtalpi_lab/temp/lmdb/valid/data.lmdb"
    output_dir = "/cto_studio/xtalpi_lab/liuzijing/ESM-Mamba/results/progen2large"

    with open("model/progen/tokenizer.json", 'r') as f:
        progen_tokenizer = Tokenizer.from_str(f.read())

    batch_size = 1
    gradient_accumulation = 1
    # breakpoint()

    ckpt_path = "/cto_studio/xtalpi_lab/liuzijing/weights/progen2-large1"
    
    model_config = ProGenConfig.from_pretrained(ckpt_path)

    model_config.torch_dtype = None
    model_config.gradient_checkpointing = True
    model_config.use_cache = False
    # model = ProGenForCausalLM(model_config)
    model = ProGenForCausalLM.from_pretrained(ckpt_path, use_cache = False, 
                                              gradient_checkpointing=True,
                                              torch_dtype=None)

    run_sm = False
    if run_sm:
        new_lm_head = nn.Linear(model.config.n_embd, 32)
        new_wte = nn.Embedding(32, model.config.n_embd)
        with torch.no_grad():
            new_lm_head.weight = nn.Parameter(model.lm_head.weight[0:32])
            new_lm_head.bias = nn.Parameter(model.lm_head.bias[0:32])
            new_wte.weight = nn.Parameter(model.transformer.wte.weight[0:32])
        model.config.vocab_size = 32
        model.transformer.vocab_size = 32
        model.lm_head = new_lm_head
        model.transformer.wte = new_wte

    for param in model.parameters():
        param.data = param.data.contiguous()

    ## freeze progen
    for param in model.transformer.parameters():
        param.requires_grad = False
    model.transformer.eval()

    gradient_checkpointing = True
    save_steps = 10
    eval_steps = 10
    save_total_limit=3

    args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        warmup_steps=5000,
        num_train_epochs=10,
        # max_steps=500000,
        learning_rate=4e-8,
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

    train_dataset = multimodal_dataset.ProgenSeqDataset(train_lmdb_path, 
                                                max_length=1024,
                                                sequence_tokenizer=progen_tokenizer)
    
    test_dataset = multimodal_dataset.ProgenSeqDataset(valid_lmdb_path, 
                                                max_length=1024,
                                                sequence_tokenizer=progen_tokenizer)


    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=multimodal_dataset.collate_fn_gpt
    )
    breakpoint()

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
