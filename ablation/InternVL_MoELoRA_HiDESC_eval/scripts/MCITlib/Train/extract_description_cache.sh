#!/bin/bash

set -euo pipefail
set -x

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

read_config_or() {
    python3 - "$1" "$2" "$3" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    config = json.load(handle)
print(config.get(sys.argv[2], sys.argv[3]))
PY
}

GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
RANK=$(read_config "$TRAIN_CONFIG" rank)
EXPERT=$(read_config "$TRAIN_CONFIG" expert_num)
MODEL_NAME=$(read_config "$MODEL_CONFIG" model_name)
MM_PROJECTOR=$(read_config "$MODEL_CONFIG" mm_projector)
DATA_PATH=$(read_config "$DATA_CONFIG" train_path)
IMAGE=$(read_config "$DATA_CONFIG" train_folder)
VISION_TOWER=$(read_config "$MODEL_CONFIG" vision_tower)
OUTPUT_DIR=$(read_config "$TRAIN_CONFIG" output_dir)
DESCRIPTION_CACHE_DIR=$(read_config_or "$TRAIN_CONFIG" description_cache_dir "$OUTPUT_DIR/description_cache")
DESCRIPTION_PROMPT=$(read_config_or "$TRAIN_CONFIG" description_prompt "Describe the image using visual evidence: objects, attributes, shapes, colors, textures, scene context, visible text, and spatial relations.")
B2_HIGH=$(read_config_or "$TRAIN_CONFIG" b2_high_layer 29)
DESCRIPTION_MAX_TOKENS=$(read_config_or "$TRAIN_CONFIG" description_max_tokens 32)
MAX_NEW=$(read_config_or "$TRAIN_CONFIG" description_cache_max_new_entries -1)
PREVIOUS=$(read_config_or "$TRAIN_CONFIG" previous_model "")

mkdir -p "$DESCRIPTION_CACHE_DIR"

LORA_ARGS=()
PREVIOUS_ARGS=()
if [[ -n "$PREVIOUS" ]]; then
    LORA_ARGS+=(--lora_enable True --lora_r "$RANK" --lora_alpha "$((RANK * 2))" --expert_num "$EXPERT")
    PREVIOUS_ARGS+=(--previous_task_model_path "$PREVIOUS")
fi

torchrun --nproc_per_node="$GPU_NUM" --master_port="${MASTER_PORT:-9001}" \
    -m llava.train.train_mem_hidesc \
    --deepspeed "${DEEPSPEED_CONFIG:-./scripts/zero2.json}" \
    "${LORA_ARGS[@]}" \
    --model_name_or_path "$MODEL_NAME" \
    "${PREVIOUS_ARGS[@]}" \
    --pretrain_mm_mlp_adapter "$MM_PROJECTOR" \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE" \
    --vision_tower "$VISION_TOWER" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -4 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir "$OUTPUT_DIR/cache_extract_runtime" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to none \
    --gradient_checkpointing False \
    --enable_description_cl False \
    --extract_description_cache_only True \
    --description_cache_dir "$DESCRIPTION_CACHE_DIR" \
    --description_prompt "$DESCRIPTION_PROMPT" \
    --b2_high_layer "$B2_HIGH" \
    --description_max_tokens "$DESCRIPTION_MAX_TOKENS" \
    --description_cache_max_new_entries "$MAX_NEW"
