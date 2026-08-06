#!/bin/bash

set -euo pipefail
set -x

################## VICUNA ##################
PROMPT_VERSION=v1
################## VICUNA ##################

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$SCRIPT_ROOT"

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

NNODES=${NNODES:-1}
MASTER_PORT=${MASTER_PORT:-9001}
GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
RANK=$(read_config "$TRAIN_CONFIG" rank)
MODEL_NAME=$(read_config "$MODEL_CONFIG" model_name)
PREVIOUS=$(read_config "$TRAIN_CONFIG" previous_model)
DATA_PATH=$(read_config "$DATA_CONFIG" train_path)
IMAGE=$(read_config "$DATA_CONFIG" train_folder)
VISION_TOWER=$(read_config "$MODEL_CONFIG" vision_tower)
OUTPUT_DIR=$(read_config "$TRAIN_CONFIG" output_dir)
EPOCH=$(read_config "$TRAIN_CONFIG" epoch)
BATCH_SIZE=$(read_config "$TRAIN_CONFIG" batch_size)
GRAD_ACC=$(read_config "$TRAIN_CONFIG" grad_acc)
LR=$(read_config "$TRAIN_CONFIG" lr)
MAX_STEPS=$(python3 -c "import json; print(json.load(open('$TRAIN_CONFIG')).get('max_steps', ''))")
SAVE_STEPS=$(python3 -c "import json; print(json.load(open('$TRAIN_CONFIG')).get('save_steps', ''))")
MODEL_MAX_LENGTH=$(python3 -c "import json; print(json.load(open('$TRAIN_CONFIG')).get('model_max_length', ''))")
DATALOADER_NUM_WORKERS=$(python3 -c "import json; print(json.load(open('$TRAIN_CONFIG')).get('dataloader_num_workers', ''))")

EXTRA_ARGS=()
if [ -n "$MAX_STEPS" ]; then EXTRA_ARGS+=(--max_steps "$MAX_STEPS"); fi
if [ -n "$SAVE_STEPS" ]; then EXTRA_ARGS+=(--save_steps "$SAVE_STEPS"); fi
if [ -n "$MODEL_MAX_LENGTH" ]; then EXTRA_ARGS+=(--model_max_length "$MODEL_MAX_LENGTH"); fi
if [ -n "$DATALOADER_NUM_WORKERS" ]; then EXTRA_ARGS+=(--dataloader_num_workers "$DATALOADER_NUM_WORKERS"); fi

mkdir -p "$OUTPUT_DIR"

GPU_LIST=""
for i in $(seq 0 $((GPU_NUM-1))); do
    GPU_LIST+="$i,"
done
GPU_LIST=${GPU_LIST%,}

echo "Begin running..."
torchrun --nnodes=${NNODES} --nproc_per_node=${GPU_NUM} --master_port "$MASTER_PORT" llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r $RANK --lora_alpha $((RANK * 2)) \
    --model_name_or_path $MODEL_NAME \
    --previous_task_model_path $PREVIOUS \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --image_folder $IMAGE \
    --vision_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -4 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps $GRAD_ACC \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    "${EXTRA_ARGS[@]}" \
    | tee ${OUTPUT_DIR}/train.log
