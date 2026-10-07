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
export MASTER_PORT="${MASTER_PORT:-29541}"

RUN_ROOT="${RUN_ROOT:-/home/lyaa/MCITlib_runs/checkpoints/MLLM-DCL/LLaVA/HiDESC/DCL_smoke_20260823}"
CACHE_ROOT="${CACHE_ROOT:-/home/lyaa/MCITlib_runs/checkpoints/MLLM-DCL/LLaVA/HiDESC/DCL_description_cache_full_20260822}"
LOG_FILE="${LOG_FILE:-/home/lyaa/MCITlib_runs/logs/mllm_dcl_hidesc_smoke_20260823.log}"
MODEL_CONFIG="${MODEL_CONFIG:-$CONFIG_ROOT/model_configs/llava.json}"

mkdir -p "$RUN_ROOT" "$(dirname "$LOG_FILE")"
if [ "${LOG_TEE_ACTIVE:-0}" != "1" ]; then
    export LOG_TEE_ACTIVE=1
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "MLLM-DCL HiDESC smoke"
echo "CUDA_VISIBLE_DEVICES=$VISIBLE_GPU_LIST"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "RUN_ROOT=$RUN_ROOT"
echo "CACHE_ROOT=$CACHE_ROOT"
echo "LOG_FILE=$LOG_FILE"

write_train_config() {
    "$PYTHON_BIN" - "$1" "$2" "$3" "$4" <<'PY'
import json
import sys

src, dst, output_dir, previous_model = sys.argv[1:]
with open(src, "r") as f:
    config = json.load(f)
config["output_dir"] = output_dir
config["gpu_num"] = 4
config["batch_size"] = 2
config["grad_acc"] = 4
config["max_steps"] = 1
config["save_steps"] = 1
config["dataloader_num_workers"] = 0
config["description_cache_model_source"] = "base"
if previous_model:
    config["previous_model"] = previous_model
elif "previous_model" in config:
    config.pop("previous_model")
with open(dst, "w") as f:
    json.dump(config, f, indent=2)
PY
}

TASK_SPECS=(
    "1 RS.json task1.json reference_description_cache_base_RS_expanded_text_v1"
    "2 Med.json task2.json reference_description_cache_base_Med_expanded_text_v1"
    "3 AD.json task3.json reference_description_cache_base_AD_expanded_text_v1"
    "4 Sci.json task4.json reference_description_cache_base_Sci_expanded_text_v1"
    "5 Fin.json task5.json reference_description_cache_base_Fin_expanded_text_v1"
)

PREVIOUS_MODEL=""
for spec in "${TASK_SPECS[@]}"; do
    read -r TASK_ID DATA_BASENAME TRAIN_BASENAME CACHE_BASENAME <<<"$spec"
    DATA_CONFIG="$CONFIG_ROOT/data_configs/MLLM-DCL/$DATA_BASENAME"
    BASE_TRAIN_CONFIG="$CONFIG_ROOT/train_configs/HiDESC/LLaVA/MLLM-DCL/train/$TRAIN_BASENAME"
    CACHE_DIR="$CACHE_ROOT/Task${TASK_ID}_llava_lora/$CACHE_BASENAME"
    TASK_OUTPUT="$RUN_ROOT/Task${TASK_ID}_llava_lora"
    TEMP_TRAIN_CONFIG="/tmp/mllm_dcl_hidesc_smoke_task${TASK_ID}_20260823.json"

    if [ ! -f "$CACHE_DIR/meta.json" ]; then
        echo "Missing completed cache metadata: $CACHE_DIR/meta.json" >&2
        exit 1
    fi

    echo "===== Task${TASK_ID}: train-only smoke ====="
    echo "Data config: $DATA_CONFIG"
    echo "Cache dir: $CACHE_DIR"
    echo "Output dir: $TASK_OUTPUT"

    write_train_config "$BASE_TRAIN_CONFIG" "$TEMP_TRAIN_CONFIG" "$TASK_OUTPUT" "$PREVIOUS_MODEL"
    "$PYTHON_BIN" - "$CACHE_DIR/meta.json" "$DATA_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    meta = json.load(f)
with open(sys.argv[2], "r") as f:
    data_cfg = json.load(f)
if meta.get("data_path") != data_cfg["train_path"]:
    raise SystemExit(
        f"Cache/data mismatch: {meta.get('data_path')} != {data_cfg['train_path']}"
    )
PY

    # Task scripts use the full train.json path from the data config, while
    # max_steps=1 keeps this invocation a fast train-only smoke test.
    export DESCRIPTION_CACHE_DIR="$CACHE_DIR"
    if [ "$TASK_ID" = "1" ]; then
        bash "$SCRIPT_DIR/Task1.sh" "$MODEL_CONFIG" "$DATA_CONFIG" "$TEMP_TRAIN_CONFIG"
    else
        bash "$SCRIPT_DIR/Taskn.sh" "$MODEL_CONFIG" "$DATA_CONFIG" "$TEMP_TRAIN_CONFIG"
    fi
    PREVIOUS_MODEL="$TASK_OUTPUT"
done

echo "MLLM-DCL HiDESC smoke completed successfully."
