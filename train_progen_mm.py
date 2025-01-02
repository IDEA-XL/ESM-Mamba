import torch 
import random
import copy
import lmdb
import os
import json

import numpy as np
from typing import Union
from tokenizers import Tokenizer

from transformers import EsmTokenizer, EsmForMaskedLM, EsmConfig
from transformers import TrainingArguments, Trainer

from transformers import LlamaForCausalLM, LlamaConfig

from model.progen.modeling_progen import ProGenForCausalLM
from model.progen.configuration_progen import ProGenConfig

from model.modeling_ss import MultiModalityCausalLM, MultiModalityConfig

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


    output_dir = "/cto_studio/xtalpi_lab/liuzijing/ESM-Mamba/results/progen2mm1"

    with open("model/progen/tokenizer.json", 'r') as f:
        progen_tokenizer = Tokenizer.from_str(f.read())

    batch_size = 64
    gradient_accumulation = 1
    # breakpoint()

    #### stage 1

    ckpt_path = "/cto_studio/xtalpi_lab/liuzijing/weights/progen2-small" 
    config_json = "/cto_studio/xtalpi_lab/liuzijing/ESM-Mamba/model/config.json"
    model_config = MultiModalityConfig.from_json_file(config_json)

    model_config.torch_dtype = None
    model_config.gradient_checkpointing = True
    model_config.use_cache = False
    model = MultiModalityCausalLM(model_config)

    model.language_model = ProGenForCausalLM.from_pretrained(ckpt_path, use_cache = False,
                                              gradient_checkpointing=True,
                                              torch_dtype=None)
    # breakpoint()

    for param in model.parameters():
        param.data = param.data.contiguous()

    ## freeze progen
    for param in model.language_model.parameters():
        param.requires_grad = False
    model.language_model.eval()

    # sum(p.numel() for p in model.parameters() if p.requires_grad) ##trainable parameters

    gradient_checkpointing = True
    save_steps = 1000
    eval_steps = 1000
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

    train_struct_name = "/cto_studio/xtalpi_lab/Datasets/AF2_ebi_processed/af_swissprot_str.pkl"
    train_dataset = multimodal_dataset.SeqStructureDataset(train_lmdb_path, train_struct_name,
                                                max_length=1024,
                                                sequence_tokenizer=progen_tokenizer)
    
    test_dataset = multimodal_dataset.SeqStructureDataset(valid_lmdb_path, train_struct_name,
                                                max_length=1024,
                                                sequence_tokenizer=progen_tokenizer)


    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=multimodal_dataset.collate_fn_mm
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
