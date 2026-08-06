#!/bin/bash

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

read_optional_config() {
    python3 - "$1" "$2" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

value = data.get(sys.argv[2], "")
print("" if value is None else value)
PY
}

GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
RANK=$(read_config "$TRAIN_CONFIG" rank)
EXPERT=$(read_config "$TRAIN_CONFIG" expert_num)
MODEL_NAME=$(read_config "$MODEL_CONFIG" model_name)
PREVIOUS=$(read_config "$TRAIN_CONFIG" previous_model)
DATA_PATH=$(read_config "$DATA_CONFIG" train_path)
IMAGE=$(read_config "$DATA_CONFIG" train_folder)
VISION_TOWER=$(read_config "$MODEL_CONFIG" vision_tower)
OUTPUT_DIR=$(read_config "$TRAIN_CONFIG" output_dir)
CUR_TASK=$(read_config "$TRAIN_CONFIG" cur_task)
EPOCH=$(read_config "$TRAIN_CONFIG" epoch)
BATCH_SIZE=$(read_config "$TRAIN_CONFIG" batch_size)
GRAD_ACC=$(read_config "$TRAIN_CONFIG" grad_acc)
LR=$(read_config "$TRAIN_CONFIG" lr)
MAX_STEPS=$(read_optional_config "$TRAIN_CONFIG" max_steps)
SAVE_STEPS=$(read_optional_config "$TRAIN_CONFIG" save_steps)
MODEL_MAX_LENGTH=$(read_optional_config "$TRAIN_CONFIG" model_max_length)
EVAL_BATCH_SIZE=$(read_optional_config "$TRAIN_CONFIG" per_device_eval_batch_size)
DATALOADER_NUM_WORKERS=$(read_optional_config "$TRAIN_CONFIG" dataloader_num_workers)
if [ -z "$SAVE_STEPS" ]; then
    SAVE_STEPS=50000
fi
if [ -z "$MODEL_MAX_LENGTH" ]; then
    MODEL_MAX_LENGTH=2048
fi
if [ -z "$EVAL_BATCH_SIZE" ]; then
    EVAL_BATCH_SIZE=16
fi
if [ -z "$DATALOADER_NUM_WORKERS" ]; then
    DATALOADER_NUM_WORKERS=4
fi

GPU_LIST=""
for i in $(seq 0 $((GPU_NUM-1))); do
    GPU_LIST+="$i,"
done
GPU_LIST=${GPU_LIST%,}

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

CMD=(
    deepspeed --include localhost:$GPU_LIST --master_port "${MASTER_PORT:-9001}" llava/train/train_mem_MOE.py
    --deepspeed ./scripts/zero2.json
    --lora_enable True --lora_r $RANK --lora_alpha $((RANK * 2)) --mm_projector_lr 2e-5
    --expert_num $EXPERT
    --model_name_or_path $MODEL_NAME
    --previous_task_model_path $PREVIOUS
    --version $PROMPT_VERSION
    --data_path $DATA_PATH
    --image_folder $IMAGE
    --vision_tower $VISION_TOWER
    --text_tower $VISION_TOWER
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio pad
    --group_by_modality_length True
    --bf16 True
    --output_dir $OUTPUT_DIR
    --cur_task $CUR_TASK
    --num_train_epochs $EPOCH
    --per_device_train_batch_size $BATCH_SIZE
    --per_device_eval_batch_size $EVAL_BATCH_SIZE
    --gradient_accumulation_steps $GRAD_ACC
    --evaluation_strategy "no"
    --save_strategy "steps"
    --save_steps $SAVE_STEPS
    --learning_rate $LR
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"
    --logging_steps 1
    --tf32 True
    --model_max_length $MODEL_MAX_LENGTH
    --gradient_checkpointing True
    --dataloader_num_workers $DATALOADER_NUM_WORKERS
    --lazy_preprocess True
    --report_to none
)

if [ -n "$MAX_STEPS" ]; then
    CMD+=(--max_steps "$MAX_STEPS")
fi

"${CMD[@]}"
