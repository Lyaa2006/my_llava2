#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_path_resolver.sh"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
CONFIG_ROOT="$MCITLIB_ROOT/configs"

cd "$PROJECT_ROOT"

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

if [ -n "${LOG_FILE:-}" ] && [ "${LOG_TEE_ACTIVE:-0}" != "1" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    export LOG_TEE_ACTIVE=1
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Logging to: $LOG_FILE"
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
VISIBLE_GPU_LIST="${CUDA_VISIBLE_DEVICES:-6,7}"
export MASTER_PORT="${MASTER_PORT:-29517}"

echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "CUDA_VISIBLE_DEVICES=$VISIBLE_GPU_LIST"
echo "MASTER_PORT=$MASTER_PORT"

MODEL_CONFIG="${1:-$CONFIG_ROOT/model_configs/llava.json}"
CACHE_ROOT="${CACHE_ROOT:-/home/lyaa/MCITlib_runs/checkpoints/MLLM-DCL/LLaVA/HiDESC/DCL_description_cache_full_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-/home/lyaa/MCITlib_runs/logs}"
mkdir -p "$CACHE_ROOT" "$LOG_DIR"

if [ ! -f "$MODEL_CONFIG" ]; then
    echo "Missing model config: $MODEL_CONFIG" >&2
    exit 1
fi

MODEL_NAME=$(read_config "$MODEL_CONFIG" model_name)
MM_PROJECTOR=$(read_config "$MODEL_CONFIG" mm_projector)
VISION_TOWER=$(read_config "$MODEL_CONFIG" vision_tower)

TASK_SPECS=(
    "1 RS.json train/task1.json"
    "2 Med.json train/task2.json"
    "3 AD.json train/task3.json"
    "4 Sci.json train/task4.json"
    "5 Fin.json train/task5.json"
)

for spec in "${TASK_SPECS[@]}"; do
    read -r TASK_ID DATA_BASENAME TRAIN_BASENAME <<<"$spec"
    DATA_CONFIG="$CONFIG_ROOT/data_configs/MLLM-DCL/$DATA_BASENAME"
    TRAIN_CONFIG="$CONFIG_ROOT/train_configs/HiDESC/LLaVA/MLLM-DCL/$TRAIN_BASENAME"

    for required_file in "$DATA_CONFIG" "$TRAIN_CONFIG"; do
        if [ ! -f "$required_file" ]; then
            echo "Missing required config file: $required_file" >&2
            exit 1
        fi
    done

    DATA_PATH=$(read_config "$DATA_CONFIG" train_path)
    IMAGE_FOLDER=$(read_config "$DATA_CONFIG" train_folder)
    OUTPUT_DIR="$CACHE_ROOT/Task${TASK_ID}_llava_lora"
    DESCRIPTION_PROMPT=$(read_optional_config "$TRAIN_CONFIG" description_prompt "Describe the image using visual evidence: objects, attributes, shapes, colors, textures, scene context, visible text, and spatial relations.")
    DESCRIPTION_MAX_TOKENS=$(read_optional_config "$TRAIN_CONFIG" description_max_tokens 32)
    DESCRIPTION_CACHE_MODEL_SOURCE=$(read_optional_config "$TRAIN_CONFIG" description_cache_model_source "base")
    CUR_TASK=$(read_config "$TRAIN_CONFIG" cur_task)
    RANK=$(read_config "$TRAIN_CONFIG" rank)
    EXPERT=$(read_config "$TRAIN_CONFIG" expert_num)
    MODEL_MAX_LENGTH=$(read_optional_config "$TRAIN_CONFIG" model_max_length 1024)
    B2_HIGH_LAYER=$(read_optional_config "$TRAIN_CONFIG" b2_high_layer 31)

    DEFAULT_CACHE_TAG=$(basename "$DATA_CONFIG" .json)
    DESCRIPTION_CACHE_DIR="$OUTPUT_DIR/reference_description_cache_${DESCRIPTION_CACHE_MODEL_SOURCE}_${DEFAULT_CACHE_TAG}_expanded_text_v1"
    quarantine_incomplete_cache_dir "$DESCRIPTION_CACHE_DIR" "Task${TASK_ID} description cache"

    echo "== Task${TASK_ID} =="
    echo "Data config: $DATA_CONFIG"
    echo "Output dir: $OUTPUT_DIR"
    echo "Description cache dir: $DESCRIPTION_CACHE_DIR"

    env -u CUDA_VISIBLE_DEVICES deepspeed --include "localhost:${VISIBLE_GPU_LIST}" --master_port "$MASTER_PORT" llava/train/train_MOE.py \
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
        --output_dir "$OUTPUT_DIR" \
        --cur_task "$CUR_TASK" \
        --model_max_length "$MODEL_MAX_LENGTH" \
        --lazy_preprocess True \
        --description_prompt "$DESCRIPTION_PROMPT" \
        --description_cache_dir "$DESCRIPTION_CACHE_DIR" \
        --description_cache_model_source "$DESCRIPTION_CACHE_MODEL_SOURCE" \
        --description_cache_max_new_entries -1 \
        --b2_high_layer "$B2_HIGH_LAYER" \
        --description_max_tokens "$DESCRIPTION_MAX_TOKENS" \
        --extract_description_cache_only True
done
