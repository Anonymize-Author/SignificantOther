#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

MODEL="./Qwen-VL/model/qwen2"  
SFT_DATA="./Qwen-VL/datasets/train_cpt"
EVAL_DATA="./Qwen-VL/datasets/dfew_39k_eval_combined_file.json"
TEST_DATA="./Qwen-VL/datasets/DFEW/dfew_test_processed_data.json"
SFT_OUT_MODEL="/mnt/bn/chenhaobo-va-data/ipr_mllm/model_results/Qwen_sft"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


torchrun $DISTRIBUTED_ARGS scripts/finetune_dfer_2.py \
    --model_name_or_path $MODEL \
    --data_path $SFT_DATA \
    --eval_data_path $EVAL_DATA \
    --bf16 True \
    --fix_vit True \
    --output_dir $SFT_OUT_MODEL \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 5 \
    --learning_rate 2e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed finetune/ds_config_zero3.json


python3 ./Qwen-VL/scripts/infre_dfer_with_cm.py --model_path $SFT_OUT_MODEL --test_data_path $TEST_DATA

