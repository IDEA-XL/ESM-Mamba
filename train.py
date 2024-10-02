import numpy as np
from typing import Union
from transformers import EsmTokenizer, EsmForMaskedLM, EsmConfig
from transformers import TrainingArguments, Trainer

from transformers import LlamaForCausalLM, LlamaConfig

import sys
sys.path.append('./utils')
from utils import multimodal_dataset

_10TB = 10995116277760

def train(ckpt):
    train_struct_name = "/cto_studio/xtalpi_lab/Datasets/af_swissprot_vqvae.pkl"

    # train_lmdb_path = "/cto_studio/xtalpi_lab/temp/lmdb/train_dedup/data.lmdb"
    # valid_lmdb_path = "/cto_studio/xtalpi_lab/temp/lmdb/valid/data.lmdb"
    # output_dir = "./results"

    train_lmdb_path = "/cto_labs/liuzijing/lmdb/train_dedup/data.lmdb"
    valid_lmdb_path = "/cto_labs/liuzijing/lmdb/valid/data.lmdb"
    output_dir = "/cto_labs/liuzijing/outputs/gpt2small_ss2"

    struct_only = False
    seq_ratio = 2
    if struct_only:
        exp_name = "struct_only"
    else:
        exp_name = "seq_struct"

    train_dataset = multimodal_dataset.SeqStructDataset(train_lmdb_path, 
                                                        train_struct_name, 
                                                        max_length=1024, seq_ratio=seq_ratio, 
                                                        struct_only=struct_only)
    
    test_dataset = multimodal_dataset.SeqStructDataset(valid_lmdb_path, 
                                                        train_struct_name, 
                                                        max_length=1024, struct_only=struct_only)

    # config = EsmConfig.from_pretrained(model_checkpoint)
    # model = EsmForMaskedLM.from_pretrained(model_checkpoint)
    # model = EsmForMaskedLM(config)
    
    configuration = LlamaConfig()
    configuration.finetuning_task = exp_name
    configuration.pad_token_id = train_dataset.sequence_tokenizer.pad_token_id
    ## 150M 
    ## progen2 small L=12 Head=16 hidden=1024; gpt2 small L=12 Head=12 hidden=768
    configuration.hidden_size = 768
    configuration.intermediate_size = 768*4
    configuration.max_position_embeddings = 1028##
    configuration.num_attention_heads = 12
    configuration.num_hidden_layers = 12
    configuration.num_key_value_heads = 12
    configuration.vocab_size = 4096 + 5 + 33##
    configuration.bos_token_id = 0

    configuration.use_cache = False

    model = LlamaForCausalLM(configuration)

    batch_size = 64
    gradient_checkpointing = True
    save_steps = 5000
    eval_steps = 5000
    save_total_limit=3


    args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1, # 2 if 4 gpus
        warmup_steps=5000,
        num_train_epochs=80,
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
        data_seed=54,
        # use_cpu=True
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
