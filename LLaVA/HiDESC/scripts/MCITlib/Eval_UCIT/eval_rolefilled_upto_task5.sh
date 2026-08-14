#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
HARD_PATH="${HARD_PATH:-$MCITLIB_ROOT}"
if [ -d "$HARD_PATH/configs" ]; then
    CONFIG_ROOT="$HARD_PATH/configs"
else
    CONFIG_ROOT="$HARD_PATH"
fi

cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
RUN_ID="${RUN_ID:-rolefilled_upto_task5_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_ID}.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

ROLEFILLED_ROOT="${ROLEFILLED_ROOT:-$MCITLIB_ROOT/checkpoints/UCIT/LLaVA/HiDESC_from_HiDeCL_from_HiDeTask1_g4b8ga2_rolefilled}"
RESULT_ROOT="${RESULT_ROOT:-$MCITLIB_ROOT/LLaVA/HiDESC/results/UCIT/full_each_dataset_hiddesc_from_hidecl_rolefilled_upto_task5}"
TEXT_TOWER="${TEXT_TOWER:-/mnt/lyaa/my_llava/clip-vit-large-patch14-336}"
MODEL_CONFIG="${MODEL_CONFIG:-$CONFIG_ROOT/modal_configs/llava.json}"
ROUTING_CONFIG_PATH="${ROUTING_CONFIG_PATH:-configs/routing_configs/HiDESC/ucit_role_3way_fft_soft.json}"
STAGE1_BAND_SCHEDULE_PATH="${STAGE1_BAND_SCHEDULE_PATH:-configs/routing_configs/HiDESC/llava_stage1_band_eval_schedule.json}"

if [ ! -d "$ROLEFILLED_ROOT" ]; then
    echo "Converted checkpoint root does not exist: $ROLEFILLED_ROOT" >&2
    exit 1
fi

ensure_existing_file "$MODEL_CONFIG" "model config"

TMP_ROOT="${TMP_ROOT:-/tmp/${RUN_ID}_eval_cfgs}"
mkdir -p "$TMP_ROOT"

echo "Logging to: $LOG_FILE"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "ROLEFILLED_ROOT=$ROLEFILLED_ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "ROUTING_CONFIG_PATH=$ROUTING_CONFIG_PATH"
echo "STAGE1_BAND_SCHEDULE_PATH=$STAGE1_BAND_SCHEDULE_PATH"
echo "TMP_ROOT=$TMP_ROOT"
echo "Start time: $(date)"

write_eval_cfg() {
    local task_id="$1"
    local cfg_path="$2"
    python3 - "$task_id" "$cfg_path" "$ROLEFILLED_ROOT" "$RESULT_ROOT" "$TEXT_TOWER" "$ROUTING_CONFIG_PATH" "$STAGE1_BAND_SCHEDULE_PATH" <<'PY'
import json
import os
import sys

task_id = int(sys.argv[1])
cfg_path = sys.argv[2]
rolefilled_root = sys.argv[3]
result_root = sys.argv[4]
text_tower = sys.argv[5]
routing_config_path = sys.argv[6]
stage1_band_schedule_path = sys.argv[7]

cfg = {
    "gpu_num": 2,
    "stage": f"HiDESC-task{task_id}-rolefilled",
    "model_path": os.path.join(rolefilled_root, f"Task{task_id}_llava_lora"),
    "result_path": result_root,
    "text_tower": text_tower,
    "num_task": 6,
    "routing_config_path": routing_config_path,
    "stage1_band_schedule_path": stage1_band_schedule_path,
}
with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(cfg_path)
PY
}

run_eval_suite() {
    local task_id="$1"
    local cfg_path="$2"

    echo
    echo "=== Evaluating Task${task_id} checkpoint on task1-${task_id} ==="
    echo "Eval config: $cfg_path"

    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh \
        "$MODEL_CONFIG" \
        "$CONFIG_ROOT/data_configs/UCIT/ImageNet-R.json" \
        "$cfg_path"

    if [ "$task_id" -ge 2 ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh \
            "$MODEL_CONFIG" \
            "$CONFIG_ROOT/data_configs/UCIT/ArxivQA.json" \
            "$cfg_path"
    fi

    if [ "$task_id" -ge 3 ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh \
            "$MODEL_CONFIG" \
            "$CONFIG_ROOT/data_configs/UCIT/VizWiz.json" \
            "$cfg_path"
    fi

    if [ "$task_id" -ge 4 ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh \
            "$MODEL_CONFIG" \
            "$CONFIG_ROOT/data_configs/UCIT/IconQA.json" \
            "$cfg_path"
    fi

    if [ "$task_id" -ge 5 ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh \
            "$MODEL_CONFIG" \
            "$CONFIG_ROOT/data_configs/UCIT/CLEVR-Math.json" \
            "$cfg_path"
    fi
}

for task_id in 2 3 4 5; do
    checkpoint_dir="$ROLEFILLED_ROOT/Task${task_id}_llava_lora"
    if [ ! -d "$checkpoint_dir" ]; then
        echo "Missing checkpoint directory: $checkpoint_dir" >&2
        exit 1
    fi
    cfg_path="$TMP_ROOT/task${task_id}.json"
    write_eval_cfg "$task_id" "$cfg_path" >/dev/null
    run_eval_suite "$task_id" "$cfg_path"
done

echo
echo "End time: $(date)"
echo "Finished task2-task5 continual evaluation."
