#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
CONFIG_ROOT="$MCITLIB_ROOT/configs"

cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export PYTHONUNBUFFERED=1
export UCIT_SMOKE_EVAL_LIMIT="${UCIT_SMOKE_EVAL_LIMIT:-4}"

COMPLETED_ROOT="${COMPLETED_ROOT:-/home/lyaa/MCITlib_runs/checkpoints/UCIT/InternVL/HiDESC/full_20260828_g45_rolebank_rebuilt_20260905_internvl4role_eval}"
RESULT_ROOT="${RESULT_ROOT:-/home/lyaa/MCITlib_runs/results/UCIT/InternVL/HiDESC/full_20260828_g45_rolebank_rebuilt_20260905_internvl4role_eval_smoke}"
MODEL_CONFIG="${MODEL_CONFIG:-$CONFIG_ROOT/model_configs/internvl.json}"
TEXT_TOWER="${TEXT_TOWER:-/mnt/lyaa/MCITlib/models/InternVL/clip-vit-large-patch14-336}"
ROUTING_CONFIG_PATH="${ROUTING_CONFIG_PATH:-$CONFIG_ROOT/routing_configs/HiDESC/ucit_role_internvl_4role_eval.json}"
STAGE1_BAND_SCHEDULE_PATH="${STAGE1_BAND_SCHEDULE_PATH:-$CONFIG_ROOT/routing_configs/HiDESC/internvl_stage1_band_eval_schedule.json}"
TASK_IDS="${TASK_IDS:-6}"
LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs}"
RUN_ID="${RUN_ID:-hidesc_offline_cache_smoke_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_ID}.log}"
TMP_ROOT="${TMP_ROOT:-/tmp/${RUN_ID}_eval_cfgs}"

mkdir -p "$LOG_DIR" "$TMP_ROOT" "$RESULT_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

for required in "$MODEL_CONFIG" "$ROUTING_CONFIG_PATH" "$STAGE1_BAND_SCHEDULE_PATH"; do
    if [ ! -f "$required" ]; then
        echo "Missing required file: $required" >&2
        exit 1
    fi
done

write_eval_config() {
    local task_id="$1"
    local cfg_path="$2"
    python3 - "$task_id" "$cfg_path" <<'PY'
import json
import os
import sys

task_id = int(sys.argv[1])
cfg_path = sys.argv[2]
cfg = {
    "gpu_num": 1,
    "stage": f"HiDESC-offline-cache-task{task_id}-smoke",
    "model_path": os.path.join(
        os.environ["COMPLETED_ROOT"], f"Task{task_id}_internvl_hidesc"
    ),
    "result_path": os.environ["RESULT_ROOT"],
    "text_tower": os.environ["TEXT_TOWER"],
    "num_task": 6,
    "routing_config_path": os.environ["ROUTING_CONFIG_PATH"],
    "stage1_band_schedule_path": os.environ["STAGE1_BAND_SCHEDULE_PATH"],
}
with open(cfg_path, "w", encoding="utf-8") as handle:
    json.dump(cfg, handle, indent=2)
PY
}

run_dataset() {
    local task_id="$1"
    local eval_cfg="$2"
    case "$task_id" in
        1)
            bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh \
                "$MODEL_CONFIG" "$CONFIG_ROOT/data_configs/UCIT/ImageNet-R-smoke.json" "$eval_cfg"
            ;;
        2)
            bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh \
                "$MODEL_CONFIG" "$CONFIG_ROOT/data_configs/UCIT/ArxivQA-smoke.json" "$eval_cfg"
            ;;
        3)
            bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh \
                "$MODEL_CONFIG" "$CONFIG_ROOT/data_configs/UCIT/VizWiz-smoke.json" "$eval_cfg"
            ;;
        4)
            bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh \
                "$MODEL_CONFIG" "$CONFIG_ROOT/data_configs/UCIT/IconQA-smoke.json" "$eval_cfg"
            ;;
        5)
            bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh \
                "$MODEL_CONFIG" "$CONFIG_ROOT/data_configs/UCIT/CLEVR-Math-smoke.json" "$eval_cfg"
            ;;
        6)
            bash scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh \
                "$MODEL_CONFIG" "$CONFIG_ROOT/data_configs/UCIT/Flickr30k-smoke.json" "$eval_cfg"
            ;;
        *)
            echo "Unsupported UCIT task id: $task_id" >&2
            exit 1
            ;;
    esac
}

export COMPLETED_ROOT RESULT_ROOT TEXT_TOWER ROUTING_CONFIG_PATH STAGE1_BAND_SCHEDULE_PATH

echo "COMPLETED_ROOT=$COMPLETED_ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "UCIT_SMOKE_EVAL_LIMIT=$UCIT_SMOKE_EVAL_LIMIT"
echo "TASK_IDS=$TASK_IDS"
echo "ROUTING_CONFIG_PATH=$ROUTING_CONFIG_PATH"
echo "STAGE1_BAND_SCHEDULE_PATH=$STAGE1_BAND_SCHEDULE_PATH"

for checkpoint_task in $TASK_IDS; do
    checkpoint_dir="$COMPLETED_ROOT/Task${checkpoint_task}_internvl_hidesc"
    if [ ! -d "$checkpoint_dir" ]; then
        echo "Missing completed checkpoint: $checkpoint_dir" >&2
        exit 1
    fi
    eval_cfg="$TMP_ROOT/task${checkpoint_task}.json"
    write_eval_config "$checkpoint_task" "$eval_cfg"
    echo
    echo "=== Task${checkpoint_task} checkpoint: evaluating task1-task${checkpoint_task} ==="
    for eval_task in $(seq 1 "$checkpoint_task"); do
        run_dataset "$eval_task" "$eval_cfg"
    done
done

echo
echo "Offline HiDESC UCIT smoke evaluation finished."
echo "Results: $RESULT_ROOT"
