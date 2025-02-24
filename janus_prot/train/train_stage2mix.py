import torch
from dataclasses import dataclass, field
from tokenizers import Tokenizer
import transformers
from transformers import TrainingArguments, Trainer

from janus_prot.model.modeling_ss import MultiModalityCausalLM
from janus_prot.data import collate_fn_slm, SeqStructureDataset, SeqStructMixDataset

local_rank = None
data_dir = "/cto_studio/xtalpi_lab/Datasets" #"/cto_labs/liuzijing/datasets/" # 
@dataclass
class DataArguments:
    train_lmdb_path: str = field(default=f"{data_dir}/lmdb/train_dedup/data.lmdb")
    valid_lmdb_path: str = field(default=f"{data_dir}/lmdb/valid/data.lmdb")
    struct2seq_path: str = field(default="/raid/swiss_prot_esm3")
    train_struct_path: list[str] = field(default_factory=lambda: [
        f"{data_dir}/AF2_ebi_processed/",
        f"{data_dir}/PDB_processed/"])
    valid_struct_path: str = field(default=f"{data_dir}/AF2_ebi_processed/UP000325664_str.pkl")

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=f"/cto_studio/xtalpi_lab/liuzijing/ESM-Mamba/results/progen2mm1/medium-pdb-checkpoint-5000")
    model_config_json: str = field(default=f"/home/liuzijing/workspace/ESM-Mamba/janus_prot/model/config_large.json")
    tokenizer_path: str = field(default=f"/cto_studio/xtalpi_lab/liuzijing/ESM-Mamba/model/progen/tokenizer.json")
    use_cache: bool = field(default=False)
    model_max_length: int = field(default=1024)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cont_ckpt: str = field(default=None)


def train():
    global local_rank

    # parse args
    parser = transformers.HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    with open(model_args.tokenizer_path, 'r') as f:
        progen_tokenizer = Tokenizer.from_str(f.read())

    model = MultiModalityCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        gradient_checkpointing=training_args.gradient_checkpointing, 
        use_cache=model_args.use_cache, 
        torch_dtype=compute_dtype
    )

    for param in model.parameters():
        param.data = param.data.contiguous()

    # init dataset
    train_dataset = SeqStructMixDataset(
        lmdb_path=data_args.train_lmdb_path, 
        struct_path=data_args.train_struct_path,
        struct2seq_path=data_args.struct2seq_path,
        max_length=model_args.model_max_length,
        seq_ratio=0.5,
        sequence_tokenizer=progen_tokenizer
    )

    valid_dataset = SeqStructureDataset(
        data_args.valid_lmdb_path,
        data_args.valid_struct_path,
        max_length=model_args.model_max_length,
        sequence_tokenizer=progen_tokenizer
    )

    # set trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn_slm
    )

    # train
    if training_args.cont_ckpt is None:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=training_args.cont_ckpt)


if __name__ == "__main__":
    train()