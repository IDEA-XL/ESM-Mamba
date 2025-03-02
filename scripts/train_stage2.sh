#!/bin/bash
# PROGEN_TYPE="large"
TOKENIZER_CONFIG_JSON=./janus_prot/model/progen/tokenizer.json
# OUT_DIR=/cto_studio/xtalpi_lab/liuzijing/ESM-Mamba/results/progen2medium_mix2
# MODEL_DIR=/cto_studio/xtalpi_lab/liuzijing/ESM-Mamba/results/progen2mix1/checkpoint-5000
OUT_DIR=/cto_studio/xtalpi_lab/liuzijing/ESM-Mamba/results/progen2medium_pdb2
MODEL_DIR=/cto_studio/xtalpi_lab/liuzijing/ESM-Mamba/results/progen2mm2medium_pdb/checkpoint-77000

deepspeed janus_prot/train/train_stage2mix.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${MODEL_DIR}\
    --tokenizer_path ${TOKENIZER_CONFIG_JSON} \
    --output_dir ${OUT_DIR} \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --use_cache False \
    --gradient_checkpointing True \
    --warmup_steps 3000 \
    --num_train_epochs 1600 \
    --optim "adamw_torch" \
    --adam_beta2 0.95 \
    --ddp_find_unused_parameters True \
    --dataloader_num_workers 2 \
    --learning_rate 1e-4 \
    --lr_scheduler_type constant_with_warmup \
    --logging_steps 1000 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 5 \
    --bf16 True \
    --report_to "tensorboard" \
    --load_best_model_at_end True \
    --seed 54