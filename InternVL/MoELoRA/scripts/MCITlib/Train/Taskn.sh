#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PROMPT_VERSION=v1

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

PYTHON_BIN="${PYTHON_BIN:-/home/lyaa/miniconda3/envs/MCITlib_copy/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/lyaa/miniconda3/envs/MCITlib_copy/bin/torchrun}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi
if [[ ! -x "$TORCHRUN_BIN" ]]; then
    TORCHRUN_BIN="$(command -v torchrun)"
fi

read_config() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json, sys
path, key = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    print(json.load(f)[key])
PY
}

read_config_default() {
    "$PYTHON_BIN" - "$1" "$2" "$3" <<'PY'
import json, sys
path, key, default = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
value = data.get(key, default)
print(value)
PY
}

NNODES=${NNODES:-1}
MASTER_PORT=${MASTER_PORT:-29511}
GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
RANK=$(read_config "$TRAIN_CONFIG" rank)
EXPERT=$(read_config "$TRAIN_CONFIG" expert_num)
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
MAX_STEPS=$(read_config_default "$TRAIN_CONFIG" max_steps -1)
SAVE_STRATEGY=$(read_config_default "$TRAIN_CONFIG" save_strategy steps)
SAVE_STEPS=$(read_config_default "$TRAIN_CONFIG" save_steps 50000)
DATA_WORKERS=$(read_config_default "$TRAIN_CONFIG" dataloader_num_workers 4)

mkdir -p "$OUTPUT_DIR"

echo "Begin running..."
"$TORCHRUN_BIN" --nnodes="${NNODES}" --nproc_per_node="${GPU_NUM}" --master_port "${MASTER_PORT}" llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r "$RANK" --lora_alpha "$((RANK * 2))" \
    --expert_num "$EXPERT" \
    --model_name_or_path "$MODEL_NAME" \
    --previous_task_model_path "$PREVIOUS" \
    --version "$PROMPT_VERSION" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE" \
    --vision_tower "$VISION_TOWER" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -4 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCH" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --evaluation_strategy "no" \
    --save_strategy "$SAVE_STRATEGY" \
    --save_steps "$SAVE_STEPS" \
    --max_steps "$MAX_STEPS" \
    --learning_rate "$LR" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers "$DATA_WORKERS" \
    --lazy_preprocess True \
    --report_to none \
    | tee "${OUTPUT_DIR}/train.log"
