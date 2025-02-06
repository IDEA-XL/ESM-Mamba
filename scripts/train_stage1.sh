#!/bin/bash
PROGEN_TYPE="medium"
PROGEN_DIR=/cto_studio/xtalpi_lab/liuzijing/weights/progen2-${PROGEN_TYPE}
MODEL_CONFIG_JSON=./janus_prot/model/config_${PROGEN_TYPE}.json
TOKENIZER_CONFIG_JSON=./janus_prot/model/progen/tokenizer.json
OUT_DIR=/cto_studio/xtalpi_lab/liuzijing/ESM-Mamba/results/progen2mix1

deepspeed janus_prot/train/train_stage1mix.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${PROGEN_DIR} \
    --tokenizer_path ${TOKENIZER_CONFIG_JSON} \
    --model_config_json ${MODEL_CONFIG_JSON} \
    --output_dir ${OUT_DIR} \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --use_cache False \
    --gradient_checkpointing True \
    --warmup_steps 500 \
    --num_train_epochs 70 \
    --optim "adamw_torch" \
    --dataloader_num_workers 0 \
    --learning_rate 1e-3 \
    --lr_scheduler_type constant_with_warmup \
    --logging_steps 1000 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --bf16 True \
    --report_to "tensorboard" \
    --load_best_model_at_end True \
    --seed 54