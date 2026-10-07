#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"

cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

BASE_TASK2_CKPT="${BASE_TASK2_CKPT:-/home/lyaa/MCITlib_runs/checkpoints/UCIT/LLaVA/HiDESC_from_task1_ce_main_g4/Task2_llava_lora_full_task1_ce_main_20260813}"
RUN_ID="${RUN_ID:-HiDESC_task3to6_from_task2_20260813_gpu01_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/home/lyaa/MCITlib_runs/checkpoints/UCIT/LLaVA/$RUN_ID}"
CFG_ROOT="${CFG_ROOT:-$RUN_ROOT/generated_configs}"
LOG_DIR="${LOG_DIR:-/home/lyaa/MCITlib_runs/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_ID.log}"

MODEL_CFG="$MCITLIB_ROOT/configs/model_configs/llava.json"

declare -A DATA_CFGS=(
    [3]="$MCITLIB_ROOT/configs/data_configs/UCIT/VizWiz.json"
    [4]="$MCITLIB_ROOT/configs/data_configs/UCIT/IconQA.json"
    [5]="$MCITLIB_ROOT/configs/data_configs/UCIT/CLEVR-Math.json"
    [6]="$MCITLIB_ROOT/configs/data_configs/UCIT/Flickr30k.json"
)

declare -A CACHE_DIRS=(
    [3]="/mnt/lyaa/MCITlib/checkpoints/UCIT/LLaVA/HiDESC_from_HiDeTask1_g4b8ga2/description_caches/task3_vizwiz"
    [4]="/mnt/lyaa/MCITlib/checkpoints/UCIT/LLaVA/HiDESC_from_HiDeTask1_g4b8ga2/description_caches/task4_iconqa"
    [5]="/mnt/lyaa/MCITlib/checkpoints/UCIT/LLaVA/HiDESC_from_HiDeTask1_g4b8ga2/description_caches/task5_clevr_math"
    [6]="/mnt/lyaa/MCITlib/checkpoints/UCIT/LLaVA/HiDESC_from_HiDeTask1_g4b8ga2/description_caches/task6_flickr30k"
)

mkdir -p "$RUN_ROOT" "$CFG_ROOT" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
set -x

if [ ! -d "$BASE_TASK2_CKPT" ]; then
    echo "Missing base task2 checkpoint: $BASE_TASK2_CKPT" >&2
    exit 1
fi

ln -sfn "$BASE_TASK2_CKPT" "$RUN_ROOT/Task2_llava_lora_base"

python3 - "$MCITLIB_ROOT" "$CFG_ROOT" "$RUN_ROOT" <<'PY'
import json
import os
import sys

mcitlib_root, cfg_root, run_root = sys.argv[1:]

templates = {
    3: os.path.join(mcitlib_root, "configs/train_configs/HiDESC/LLaVA/UCIT/train/task3.json"),
    4: os.path.join(mcitlib_root, "configs/train_configs/HiDESC/LLaVA/UCIT/train/task4.json"),
    5: os.path.join(mcitlib_root, "configs/train_configs/HiDESC/LLaVA/UCIT/train/task5.json"),
    6: os.path.join(mcitlib_root, "configs/train_configs/HiDESC/LLaVA/UCIT/train/task6.json"),
}

cache_dirs = {
    3: "/mnt/lyaa/MCITlib/checkpoints/UCIT/LLaVA/HiDESC_from_HiDeTask1_g4b8ga2/description_caches/task3_vizwiz",
    4: "/mnt/lyaa/MCITlib/checkpoints/UCIT/LLaVA/HiDESC_from_HiDeTask1_g4b8ga2/description_caches/task4_iconqa",
    5: "/mnt/lyaa/MCITlib/checkpoints/UCIT/LLaVA/HiDESC_from_HiDeTask1_g4b8ga2/description_caches/task5_clevr_math",
    6: "/mnt/lyaa/MCITlib/checkpoints/UCIT/LLaVA/HiDESC_from_HiDeTask1_g4b8ga2/description_caches/task6_flickr30k",
}

previous = os.path.join(run_root, "Task2_llava_lora_base")
for tid in (3, 4, 5, 6):
    with open(templates[tid], "r") as f:
        cfg = json.load(f)
    cfg["previous_model"] = previous
    cfg["output_dir"] = os.path.join(run_root, f"Task{tid}_llava_lora")
    cfg["description_cache_dir"] = cache_dirs[tid]
    out_path = os.path.join(cfg_root, f"train_task{tid}.json")
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    previous = cfg["output_dir"]
PY

for tid in 3 4 5 6; do
    if [ ! -d "${CACHE_DIRS[$tid]}" ]; then
        echo "Missing description cache for task${tid}: ${CACHE_DIRS[$tid]}" >&2
        exit 1
    fi
done

echo "RUN_ID=$RUN_ID"
echo "RUN_ROOT=$RUN_ROOT"
echo "CFG_ROOT=$CFG_ROOT"
echo "LOG_FILE=$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "BASE_TASK2_CKPT=$BASE_TASK2_CKPT"

for tid in 3 4 5 6; do
    echo "===== Train task${tid} ====="
    bash scripts/MCITlib/Train/Taskn.sh \
        "$MODEL_CFG" \
        "${DATA_CFGS[$tid]}" \
        "$CFG_ROOT/train_task${tid}.json"
done

echo "===== Done ====="
echo "checkpoints: $RUN_ROOT"
echo "log: $LOG_FILE"
