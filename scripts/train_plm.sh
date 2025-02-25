#!/bin/bash
PROGEN_TYPE="small"
PROGEN_DIR=/cto_labs/liuzijing/weights/progen2-${PROGEN_TYPE}
MODEL_CONFIG_JSON=./janus_prot/model/config_plm_${PROGEN_TYPE}.json
TOKENIZER_CONFIG_JSON=./janus_prot/model/progen/tokenizer.json
OUT_DIR=/cto_labs/liuzijing/outputs/plm_${PROGEN_TYPE}

deepspeed janus_prot/train/train_plm.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${PROGEN_DIR} \
    --tokenizer_path ${TOKENIZER_CONFIG_JSON} \
    --model_config_json ${MODEL_CONFIG_JSON} \
    --output_dir ${OUT_DIR} \
    --per_device_train_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --use_cache False \
    --gradient_checkpointing True \
    --warmup_steps 5000 \
    --num_train_epochs 10 \
    --optim "adamw_torch" \
    --dataloader_num_workers 2 \
    --learning_rate 6e-4 \
    --lr_scheduler_type cosine \
    --logging_steps 2000 \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 5 \
    --bf16 True \
    --report_to "tensorboard" \
    --load_best_model_at_end True \
    --seed 54