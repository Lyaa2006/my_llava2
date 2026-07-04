#!/bin/bash
set -e

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
    python3 - "$1" "$2" "$3" <<'PY'
import json
import sys

path, key, default = sys.argv[1:]
with open(path, "r") as f:
    data = json.load(f)
value = data.get(key, default)
if isinstance(value, bool):
    print("True" if value else "False")
else:
    print(value)
PY
}

count_expected_cache_entries() {
    python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    data = json.load(f)
print(sum(1 for sample in data if "image" in sample))
PY
}

cache_meta_matches() {
    python3 - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json
import os
import sys

cache_dir, data_path, prompt, hidden_layer, max_tokens = sys.argv[1:]
meta_path = os.path.join(cache_dir, "meta.json")
if not os.path.exists(meta_path):
    print("False")
    raise SystemExit
with open(meta_path, "r") as f:
    meta = json.load(f)
ok = (
    meta.get("data_path") == data_path
    and meta.get("description_prompt") == prompt
    and str(meta.get("description_hidden_layer")) == str(hidden_layer)
    and str(meta.get("description_max_tokens")) == str(max_tokens)
)
print("True" if ok else "False")
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
RUN_SUFFIX=${UCIT_RUN_ID:+_$UCIT_RUN_ID}
OUTPUT_DIR="${OUTPUT_DIR}${RUN_SUFFIX}"

DESCRIPTION_PROMPT=$(read_optional_config "$TRAIN_CONFIG" description_prompt "Describe the image using visual evidence: objects, attributes, shapes, colors, textures, scene context, visible text, and spatial relations.")
DESCRIPTION_HIDDEN_LAYER=$(read_optional_config "$TRAIN_CONFIG" description_hidden_layer -2)
DESCRIPTION_MAX_TOKENS=$(read_optional_config "$TRAIN_CONFIG" description_max_tokens 32)
DESCRIPTION_ALIGN_WEIGHT=$(read_optional_config "$TRAIN_CONFIG" description_align_weight 1.0)
DESCRIPTION_UTILITY_WEIGHT=$(read_optional_config "$TRAIN_CONFIG" description_utility_weight 1.0)
STANDARD_CE_WEIGHT=$(read_optional_config "$TRAIN_CONFIG" standard_ce_weight 1.0)
SAVE_STEPS=$(read_optional_config "$TRAIN_CONFIG" save_steps 50000)
MODEL_MAX_LENGTH=$(read_optional_config "$TRAIN_CONFIG" model_max_length 2048)
DATALOADER_NUM_WORKERS=$(read_optional_config "$TRAIN_CONFIG" dataloader_num_workers 4)
MAX_STEPS=$(read_optional_config "$TRAIN_CONFIG" max_steps -1)

DEFAULT_CACHE_TAG=$(basename "$DATA_PATH" .json)
DEFAULT_DESCRIPTION_CACHE_DIR="$PREVIOUS/reference_description_cache_${DEFAULT_CACHE_TAG}"
DESCRIPTION_CACHE_DIR="${DESCRIPTION_CACHE_DIR:-$(read_optional_config "$TRAIN_CONFIG" description_cache_dir "$DEFAULT_DESCRIPTION_CACHE_DIR")}"

if [ ! -d "$PREVIOUS" ]; then
    echo "Previous task checkpoint does not exist: $PREVIOUS" >&2
    exit 1
fi

echo "Previous checkpoint: $PREVIOUS"
echo "Output checkpoint: $OUTPUT_DIR"
echo "Description cache dir: $DESCRIPTION_CACHE_DIR"

EXPECTED_CACHE_ENTRIES=$(count_expected_cache_entries "$DATA_PATH")
CACHE_READY=False

if [ -d "$DESCRIPTION_CACHE_DIR" ]; then
    EXISTING_CACHE_ENTRIES=$(find "$DESCRIPTION_CACHE_DIR" -maxdepth 1 -name '*.pt' | wc -l)
    CACHE_META_READY=$(cache_meta_matches \
        "$DESCRIPTION_CACHE_DIR" \
        "$DATA_PATH" \
        "$DESCRIPTION_PROMPT" \
        "$DESCRIPTION_HIDDEN_LAYER" \
        "$DESCRIPTION_MAX_TOKENS")
    if [ "$EXISTING_CACHE_ENTRIES" -ge "$EXPECTED_CACHE_ENTRIES" ] && [ "$EXPECTED_CACHE_ENTRIES" -gt 0 ] && [ "$CACHE_META_READY" = "True" ]; then
        CACHE_READY=True
    elif [ "$EXISTING_CACHE_ENTRIES" -gt 0 ]; then
        echo "Description cache exists but metadata does not match current prompt/settings; rebuilding: $DESCRIPTION_CACHE_DIR"
    fi
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_LIST="$CUDA_VISIBLE_DEVICES"
    GPU_NUM=$(python3 -c "print(len([x for x in '${CUDA_VISIBLE_DEVICES}'.split(',') if x.strip()]))")
else
    GPU_LIST=""
    for i in $(seq 0 $((GPU_NUM-1))); do
        GPU_LIST+="$i,"
    done
    GPU_LIST=${GPU_LIST%,}
fi

if [ -z "${MASTER_PORT:-}" ]; then
    MASTER_PORT=$(python3 - <<'PY'
import socket

s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)
fi

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

EXTRA_ARGS=""
if [ "$MAX_STEPS" -gt 0 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --max_steps $MAX_STEPS"
fi

if [ "$CACHE_READY" != "True" ]; then
    rm -rf "$DESCRIPTION_CACHE_DIR"
    python llava/train/train_MOE.py \
        --lora_enable True \
        --lora_r $RANK \
        --lora_alpha $((RANK * 2)) \
        --expert_num $EXPERT \
        --model_name_or_path $MODEL_NAME \
        --previous_task_model_path $PREVIOUS \
        --version $PROMPT_VERSION \
        --data_path $DATA_PATH \
        --image_folder $IMAGE \
        --vision_tower $VISION_TOWER \
        --text_tower $VISION_TOWER \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --bf16 True \
        --output_dir $OUTPUT_DIR \
        --cur_task $CUR_TASK \
        --model_max_length $MODEL_MAX_LENGTH \
        --lazy_preprocess True \
        --description_prompt "$DESCRIPTION_PROMPT" \
        --description_cache_dir "$DESCRIPTION_CACHE_DIR" \
        --description_hidden_layer $DESCRIPTION_HIDDEN_LAYER \
        --description_max_tokens $DESCRIPTION_MAX_TOKENS \
        --extract_description_cache_only True
fi

deepspeed --include localhost:$GPU_LIST --master_port "${MASTER_PORT:-9001}" llava/train/train_mem_MOE.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r $RANK --lora_alpha $((RANK * 2)) --mm_projector_lr 2e-5 \
    --expert_num $EXPERT \
    --model_name_or_path $MODEL_NAME \
    --previous_task_model_path $PREVIOUS \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --image_folder $IMAGE \
    --vision_tower $VISION_TOWER \
    --text_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --cur_task $CUR_TASK \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps $GRAD_ACC \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --lazy_preprocess True \
    --description_prompt "$DESCRIPTION_PROMPT" \
    --description_cache_dir "$DESCRIPTION_CACHE_DIR" \
    --enable_description_cl True \
    --description_hidden_layer $DESCRIPTION_HIDDEN_LAYER \
    --description_max_tokens $DESCRIPTION_MAX_TOKENS \
    --description_align_weight $DESCRIPTION_ALIGN_WEIGHT \
    --description_utility_weight $DESCRIPTION_UTILITY_WEIGHT \
    --standard_ce_weight $STANDARD_CE_WEIGHT \
    --report_to none \
    $EXTRA_ARGS
