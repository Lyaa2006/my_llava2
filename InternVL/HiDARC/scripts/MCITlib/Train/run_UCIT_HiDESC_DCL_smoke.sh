#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_path_resolver.sh"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
CONFIG_ROOT="$MCITLIB_ROOT/configs"

cd "$PROJECT_ROOT"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
VISIBLE_GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MASTER_PORT="${MASTER_PORT:-29531}"

RUN_ROOT="${RUN_ROOT:-/home/lyaa/MCITlib_runs/checkpoints/UCIT/LLaVA/HiDESC/DCL_smoke_20260823}"
CACHE_ROOT="${CACHE_ROOT:-$RUN_ROOT/description_caches}"
LOG_FILE="${LOG_FILE:-/home/lyaa/MCITlib_runs/logs/ucit_hidesc_dcl_smoke_20260823.log}"
MODEL_CONFIG="${MODEL_CONFIG:-$CONFIG_ROOT/model_configs/llava.json}"

mkdir -p "$RUN_ROOT" "$CACHE_ROOT" "$(dirname "$LOG_FILE")"
if [ "${LOG_TEE_ACTIVE:-0}" != "1" ]; then
    export LOG_TEE_ACTIVE=1
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "UCIT HiDESC+DCL smoke"
echo "CUDA_VISIBLE_DEVICES=$VISIBLE_GPU_LIST"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "RUN_ROOT=$RUN_ROOT"
echo "CACHE_ROOT=$CACHE_ROOT"
echo "LOG_FILE=$LOG_FILE"

read_json() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    value = json.load(f)[sys.argv[2]]
print(value)
PY
}

read_optional_json() {
    "$PYTHON_BIN" - "$1" "$2" "$3" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    value = json.load(f).get(sys.argv[2], sys.argv[3])
print(value)
PY
}

write_train_config() {
    "$PYTHON_BIN" - "$1" "$2" "$3" "$4" "$5" "$6" <<'PY'
import json
import sys

src, dst, output_dir, previous_model, cache_dir, task_id = sys.argv[1:]
with open(src, "r") as f:
    config = json.load(f)
config["output_dir"] = output_dir
config["description_cache_dir"] = cache_dir
config["description_cache_model_source"] = "base"
config["max_steps"] = 1
config["save_steps"] = 1
config["batch_size"] = 1
config["grad_acc"] = 2
config["gpu_num"] = 4
if previous_model:
    config["previous_model"] = previous_model
elif "previous_model" in config:
    config.pop("previous_model")
with open(dst, "w") as f:
    json.dump(config, f, indent=2)
PY
}

MODEL_NAME=$(read_json "$MODEL_CONFIG" model_name)
MM_PROJECTOR=$(read_json "$MODEL_CONFIG" mm_projector)
VISION_TOWER=$(read_json "$MODEL_CONFIG" vision_tower)

TASK_SPECS=(
    "1 ImageNet-R-smoke.json task1_smoke.json"
    "2 ArxivQA-smoke.json task2_smoke.json"
    "3 VizWiz-smoke.json task3_smoke.json"
    "4 IconQA-smoke.json task4_smoke.json"
    "5 CLEVR-Math-smoke.json task5_smoke.json"
    "6 Flickr30k-smoke.json task6_smoke.json"
)

PREVIOUS_MODEL=""
for spec in "${TASK_SPECS[@]}"; do
    read -r TASK_ID DATA_BASENAME TRAIN_BASENAME <<<"$spec"
    DATA_CONFIG="$CONFIG_ROOT/data_configs/UCIT/$DATA_BASENAME"
    BASE_TRAIN_CONFIG="$CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/train/$TRAIN_BASENAME"
    TEMP_TRAIN_CONFIG="/tmp/ucit_hidesc_dcl_smoke_task${TASK_ID}_20260823.json"
    TASK_OUTPUT="$RUN_ROOT/Task${TASK_ID}_llava_lora"
    DATA_TAG="${DATA_BASENAME%.json}"
    CACHE_DIR="$CACHE_ROOT/Task${TASK_ID}_llava_lora/reference_description_cache_base_${DATA_TAG}_expanded_text_v1"

    DATA_PATH=$(read_json "$DATA_CONFIG" train_path)
    IMAGE_FOLDER=$(read_json "$DATA_CONFIG" train_folder)
    CUR_TASK=$(read_json "$BASE_TRAIN_CONFIG" cur_task)
    RANK=$(read_json "$BASE_TRAIN_CONFIG" rank)
    EXPERT=$(read_json "$BASE_TRAIN_CONFIG" expert_num)
    DESCRIPTION_PROMPT=$(read_optional_json "$BASE_TRAIN_CONFIG" description_prompt "Describe the image using visual evidence: objects, attributes, shapes, colors, textures, scene context, visible text, and spatial relations.")
    DESCRIPTION_MAX_TOKENS=$(read_optional_json "$BASE_TRAIN_CONFIG" description_max_tokens 32)
    B2_HIGH_LAYER=$(read_optional_json "$BASE_TRAIN_CONFIG" b2_high_layer 31)

    echo "===== Task${TASK_ID}: extracting UCIT smoke cache ====="
    echo "Data: $DATA_PATH"
    echo "Cache: $CACHE_DIR"

    env -u CUDA_VISIBLE_DEVICES "$DEEPSPEED_BIN" --include "localhost:${VISIBLE_GPU_LIST}" --master_port "$MASTER_PORT" llava/train/train_MOE.py \
        --lora_enable True \
        --lora_r "$RANK" \
        --lora_alpha "$((RANK * 2))" \
        --expert_num "$EXPERT" \
        --model_name_or_path "$MODEL_NAME" \
        --pretrain_mm_mlp_adapter "$MM_PROJECTOR" \
        --version v1 \
        --data_path "$DATA_PATH" \
        --image_folder "$IMAGE_FOLDER" \
        --vision_tower "$VISION_TOWER" \
        --text_tower "$VISION_TOWER" \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --bf16 True \
        --output_dir "$TASK_OUTPUT" \
        --cur_task "$CUR_TASK" \
        --model_max_length 1024 \
        --lazy_preprocess True \
        --description_prompt "$DESCRIPTION_PROMPT" \
        --description_cache_dir "$CACHE_DIR" \
        --description_cache_model_source base \
        --description_cache_max_new_entries -1 \
        --b2_high_layer "$B2_HIGH_LAYER" \
        --description_max_tokens "$DESCRIPTION_MAX_TOKENS" \
        --extract_description_cache_only True

    write_train_config \
        "$BASE_TRAIN_CONFIG" \
        "$TEMP_TRAIN_CONFIG" \
        "$TASK_OUTPUT" \
        "$PREVIOUS_MODEL" \
        "$CACHE_DIR" \
        "$TASK_ID"

    echo "===== Task${TASK_ID}: train-only smoke ====="
    if [ "$TASK_ID" = "1" ]; then
        bash "$SCRIPT_DIR/Task1.sh" "$MODEL_CONFIG" "$DATA_CONFIG" "$TEMP_TRAIN_CONFIG"
    else
        bash "$SCRIPT_DIR/Taskn.sh" "$MODEL_CONFIG" "$DATA_CONFIG" "$TEMP_TRAIN_CONFIG"
    fi
    PREVIOUS_MODEL="$TASK_OUTPUT"
done

echo "UCIT HiDESC+DCL smoke completed successfully."
