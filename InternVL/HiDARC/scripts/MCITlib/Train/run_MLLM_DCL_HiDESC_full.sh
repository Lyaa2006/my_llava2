#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_path_resolver.sh"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
CONFIG_ROOT="$MCITLIB_ROOT/configs"

cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export MASTER_PORT="${MASTER_PORT:-29543}"

RUN_ROOT="${RUN_ROOT:-/home/lyaa/MCITlib_runs/checkpoints/MLLM-DCL/LLaVA/HiDESC/DCL_full_20260823}"
CACHE_ROOT="${CACHE_ROOT:-/home/lyaa/MCITlib_runs/checkpoints/MLLM-DCL/LLaVA/HiDESC/DCL_description_cache_full_20260822}"
LOG_FILE="${LOG_FILE:-/home/lyaa/MCITlib_runs/logs/mllm_dcl_hidesc_full_20260823.log}"
MODEL_CONFIG="${MODEL_CONFIG:-$CONFIG_ROOT/model_configs/llava.json}"

mkdir -p "$RUN_ROOT" "$(dirname "$LOG_FILE")"
if [ "${LOG_TEE_ACTIVE:-0}" != "1" ]; then
    export LOG_TEE_ACTIVE=1
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "MLLM-DCL HiDESC full train-only run"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "MASTER_PORT=$MASTER_PORT"
echo "RUN_ROOT=$RUN_ROOT"
echo "CACHE_ROOT=$CACHE_ROOT"
echo "LOG_FILE=$LOG_FILE"
echo "global_batch=4x2x4=32"

write_train_config() {
    "$PYTHON_BIN" - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json
import sys

src, dst, output_dir, previous_model, cache_dir = sys.argv[1:]
with open(src, "r") as f:
    config = json.load(f)

config["output_dir"] = output_dir
config["gpu_num"] = 4
config["batch_size"] = 2
config["grad_acc"] = 4
config["description_cache_model_source"] = "base"
config["description_cache_dir"] = cache_dir
config.pop("max_steps", None)
if previous_model:
    config["previous_model"] = previous_model
else:
    config.pop("previous_model", None)

with open(dst, "w") as f:
    json.dump(config, f, indent=2)
PY
}

validate_cache() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
import os
import sys

cache_dir, data_config_path = sys.argv[1:]
meta_path = os.path.join(cache_dir, "meta.json")
if not os.path.isfile(meta_path):
    raise SystemExit(f"Missing cache metadata: {meta_path}")

with open(meta_path, "r") as f:
    meta = json.load(f)
with open(data_config_path, "r") as f:
    data_config = json.load(f)
with open(data_config["train_path"], "r") as f:
    samples = json.load(f)

expected = sum(1 for sample in samples if "image" in sample)
actual = sum(1 for name in os.listdir(cache_dir) if name.endswith(".pt"))
if meta.get("data_path") != data_config["train_path"]:
    raise SystemExit(
        f"Cache/data mismatch: {meta.get('data_path')} != {data_config['train_path']}"
    )
if meta.get("description_cache_model_source", "previous") != "base":
    raise SystemExit(f"Cache source is not base: {meta.get('description_cache_model_source')}")
if meta.get("description_cache_format") != "expanded_text_v1":
    raise SystemExit(f"Unexpected cache format: {meta.get('description_cache_format')}")
if int(meta.get("b2_high_layer", -1)) != 31:
    raise SystemExit(f"Unexpected b2_high_layer: {meta.get('b2_high_layer')}")
if int(meta.get("description_max_tokens", -1)) != 32:
    raise SystemExit(f"Unexpected description_max_tokens: {meta.get('description_max_tokens')}")
if actual < expected:
    raise SystemExit(f"Incomplete cache: {actual}/{expected} entries in {cache_dir}")
print(f"Cache validated: {actual}/{expected} entries, {cache_dir}")
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
    read -r TASK_ID DATA_BASENAME TRAIN_BASENAME CACHE_BASENAME <<< "$spec"
    DATA_CONFIG="$CONFIG_ROOT/data_configs/MLLM-DCL/$DATA_BASENAME"
    BASE_TRAIN_CONFIG="$CONFIG_ROOT/train_configs/HiDESC/LLaVA/MLLM-DCL/train/$TRAIN_BASENAME"
    CACHE_DIR="$CACHE_ROOT/Task${TASK_ID}_llava_lora/$CACHE_BASENAME"
    TASK_OUTPUT="$RUN_ROOT/Task${TASK_ID}_llava_lora"
    TEMP_TRAIN_CONFIG="/tmp/mllm_dcl_hidesc_full_task${TASK_ID}_20260823.json"

    validate_cache "$CACHE_DIR" "$DATA_CONFIG"
    write_train_config "$BASE_TRAIN_CONFIG" "$TEMP_TRAIN_CONFIG" "$TASK_OUTPUT" "$PREVIOUS_MODEL" "$CACHE_DIR"

    echo "===== Task${TASK_ID}: full train-only ====="
    echo "Data config: $DATA_CONFIG"
    echo "Cache dir: $CACHE_DIR"
    echo "Output dir: $TASK_OUTPUT"
    echo "Previous model: ${PREVIOUS_MODEL:-<base model>}"

    export DESCRIPTION_CACHE_DIR="$CACHE_DIR"
    if [ "$TASK_ID" = "1" ]; then
        bash "$SCRIPT_DIR/Task1.sh" "$MODEL_CONFIG" "$DATA_CONFIG" "$TEMP_TRAIN_CONFIG"
    else
        bash "$SCRIPT_DIR/Taskn.sh" "$MODEL_CONFIG" "$DATA_CONFIG" "$TEMP_TRAIN_CONFIG"
    fi
    PREVIOUS_MODEL="$TASK_OUTPUT"
done

echo "MLLM-DCL HiDESC full train-only run completed successfully."
