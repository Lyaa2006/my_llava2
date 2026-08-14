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
MODEL_NAME=$(read_config "$MODEL_CONFIG" model_name)
MM_PROJECTOR=$(read_config "$MODEL_CONFIG" mm_projector)
PREVIOUS=$(read_config "$TRAIN_CONFIG" previous_model)
DATA_PATH=$(read_config "$DATA_CONFIG" train_path)
IMAGE=$(read_config "$DATA_CONFIG" train_folder)
VISION_TOWER=$(read_config "$MODEL_CONFIG" vision_tower)
OUTPUT_DIR=$(read_config "$TRAIN_CONFIG" output_dir)
EPOCH=$(read_config "$TRAIN_CONFIG" epoch)
BATCH_SIZE=$(read_config "$TRAIN_CONFIG" batch_size)
GRAD_ACC=$(read_config "$TRAIN_CONFIG" grad_acc)
LR=$(read_config "$TRAIN_CONFIG" lr)

DESCRIPTION_CACHE_DIR=$(read_config_or "$TRAIN_CONFIG" description_cache_dir "$OUTPUT_DIR/description_cache")
B1_LOW=$(read_config_or "$TRAIN_CONFIG" b1_low_layer 12)
B1_HIGH=$(read_config_or "$TRAIN_CONFIG" b1_high_layer 14)
B1_CENTER=$(read_config_or "$TRAIN_CONFIG" b1_center_layer 14)
B2_LOW=$(read_config_or "$TRAIN_CONFIG" b2_low_layer 27)
B2_HIGH=$(read_config_or "$TRAIN_CONFIG" b2_high_layer 29)
B2_CENTER=$(read_config_or "$TRAIN_CONFIG" b2_center_layer 28)
DESCRIPTION_MAX_TOKENS=$(read_config_or "$TRAIN_CONFIG" description_max_tokens 32)
DESCRIPTION_FOCUS_WEIGHT=$(read_config_or "$TRAIN_CONFIG" description_focus_weight 0.2)
DESCRIPTION_ENERGY_WEIGHT=$(read_config_or "$TRAIN_CONFIG" description_energy_weight 0.0001)
DESCRIPTION_ENERGY_MARGIN=$(read_config_or "$TRAIN_CONFIG" description_energy_margin 30.0)
ALIGN_LOSS_WEIGHT=$(read_config_or "$TRAIN_CONFIG" align_loss_weight 0.01)

mkdir -p "$OUTPUT_DIR"

torchrun --nproc_per_node="$GPU_NUM" --master_port="${MASTER_PORT:-9001}" \
    -m llava.train.train_mem_hidesc_lora \
    --deepspeed "${DEEPSPEED_CONFIG:-./scripts/zero2.json}" \
    --lora_enable True --lora_r "$RANK" --lora_alpha "$((RANK * 2))" \
    --model_name_or_path "$MODEL_NAME" \
    --previous_task_model_path "$PREVIOUS" \
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
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCH" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 50000 \
    --learning_rate "$LR" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --enable_description_cl True \
    --description_cache_dir "$DESCRIPTION_CACHE_DIR" \
    --b1_low_layer "$B1_LOW" \
    --b1_high_layer "$B1_HIGH" \
    --b1_center_layer "$B1_CENTER" \
    --b2_low_layer "$B2_LOW" \
    --b2_high_layer "$B2_HIGH" \
    --b2_center_layer "$B2_CENTER" \
    --description_max_tokens "$DESCRIPTION_MAX_TOKENS" \
    --description_focus_weight "$DESCRIPTION_FOCUS_WEIGHT" \
    --description_energy_weight "$DESCRIPTION_ENERGY_WEIGHT" \
    --description_energy_margin "$DESCRIPTION_ENERGY_MARGIN" \
    --align_loss_weight "$ALIGN_LOSS_WEIGHT" \
    | tee "$OUTPUT_DIR/train_hidesc_lora.log"
