import torch
from dataclasses import dataclass, field
from tokenizers import Tokenizer
import transformers
from transformers import TrainingArguments, Trainer

from transformers import LlamaForCausalLM, LlamaConfig

from janus_prot.data import collate_fn_gpt, ProgenSeqDataset

local_rank = None
data_dir = "/cto_labs/liuzijing/datasets/" # "/cto_studio/xtalpi_lab/Datasets" #

@dataclass
class DataArguments:
    train_lmdb_path: str = field(default=f"{data_dir}/lmdb/train_dedup/data.lmdb")
    valid_lmdb_path: str = field(default=f"{data_dir}/lmdb/valid/data.lmdb")

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=f"/cto_studio/xtalpi_lab/liuzijing/weights/progen2-small")
    model_config_json: str = field(default=f"/cto_studio/xtalpi_lab/liuzijing/ESM-Mamba/model/config_small.json")
    tokenizer_path: str = field(default=f"/cto_studio/xtalpi_lab/liuzijing/ESM-Mamba/model/progen/tokenizer.json")
    use_cache: bool = field(default=False)
    model_max_length: int = field(default=1024)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cont_ckpt: str = field(default=None, metadata={"help": "Continue training from checkpoint"})


def train():
    global local_rank

    # parse args
    parser = transformers.HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    with open(model_args.tokenizer_path, 'r') as f:
        progen_tokenizer = Tokenizer.from_str(f.read())

    # init model config
    model_config = LlamaConfig.from_json_file(model_args.model_config_json)
    # override model config
    model_config.torch_dtype = compute_dtype
    model_config.gradient_checkpointing = training_args.gradient_checkpointing
    model_config.use_cache = model_args.use_cache
    # init model
    model = LlamaForCausalLM(model_config)

    print("total parameters", sum(p.numel() for p in model.parameters() if p.requires_grad)) ##trainable parameters

    # init dataset
    train_dataset = ProgenSeqDataset(
        lmdb_path=data_args.train_lmdb_path, 
        max_length=model_args.model_max_length,
        sequence_tokenizer=progen_tokenizer
    )

    valid_dataset = ProgenSeqDataset(
        lmdb_path=data_args.valid_lmdb_path,
        max_length=model_args.model_max_length,
        sequence_tokenizer=progen_tokenizer
    )

    # set trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn_gpt
    )
    
    # train
    if training_args.cont_ckpt is None:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=training_args.cont_ckpt)


if __name__ == "__main__":
    train()