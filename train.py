import numpy as np
from typing import Union
from transformers import EsmTokenizer, EsmForMaskedLM, EsmConfig
from transformers import TrainingArguments, Trainer

from transformers import LlamaForCausalLM, LlamaConfig

import sys
sys.path.append('./utils')
from utils import multimodal_dataset

_10TB = 10995116277760

def train():
    train_struct_name = "/cto_studio/xtalpi_lab/Datasets/af_swissprot_vqvae.pkl"
    train_lmdb_path = "/cto_studio/xtalpi_lab/temp/lmdb/train_dedup/data.lmdb"
    valid_lmdb_path = "/cto_studio/xtalpi_lab/temp/lmdb/valid/data.lmdb"

    train_dataset = multimodal_dataset.SeqStructDataset(valid_lmdb_path, 
                                                        train_struct_name, max_length=1024)
    
    test_dataset = multimodal_dataset.SeqStructDataset(valid_lmdb_path, 
                                                        train_struct_name, max_length=512)

    # config = EsmConfig.from_pretrained(model_checkpoint)
    # model = EsmForMaskedLM.from_pretrained(model_checkpoint)
    # model = EsmForMaskedLM(config)
    
    configuration = LlamaConfig()
    ## 150M
    configuration.hidden_size = 640
    configuration.intermediate_size = 2560
    configuration.max_position_embeddings = 1028##
    configuration.num_attention_heads = 20
    configuration.num_hidden_layers = 30
    configuration.num_key_value_heads = 20
    configuration.vocab_size = 4096 + 5 + 33##
    configuration.bos_token_id = 0

    configuration.use_cache = False

    model = LlamaForCausalLM(configuration)

    batch_size = 16
    gradient_checkpointing = True
    save_steps = 5000
    eval_steps = 5000
    save_total_limit=3
    output_dir = "./results"


    args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
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

    trainer.train()


if __name__ == "__main__":
    train()
