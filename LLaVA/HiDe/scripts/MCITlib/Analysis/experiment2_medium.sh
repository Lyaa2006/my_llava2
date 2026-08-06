#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"

cd "$PROJECT_ROOT"

if [ -x "/home/lyaa/miniconda3/envs/MCITlib/bin/python3.1" ]; then
    PYTHON_BIN="${PYTHON_BIN:-/home/lyaa/miniconda3/envs/MCITlib/bin/python3.1}"
elif [ -x "/home/lyaa/miniconda3/envs/MCITlib/bin/python3" ]; then
    PYTHON_BIN="${PYTHON_BIN:-/home/lyaa/miniconda3/envs/MCITlib/bin/python3}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/lyaa/my_llava/llava-v1.5-7b}"
VISION_TOWER_PATH="${VISION_TOWER_PATH:-/mnt/lyaa/my_llava/clip-vit-large-patch14-336}"
UCIT_ROOT="${UCIT_ROOT:-/mnt/lyaa/my_llava/UCIT}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe}"
OUTPUT_DIR="${OUTPUT_DIR:-$MCITLIB_ROOT/docs/experiment2_outputs_medium}"
TASK_IDS="${TASK_IDS:-1 2 3 4 5 6}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-256}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cuda}"

LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/experiment2_l_focus_medium_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1

read -r -a TASK_ID_ARRAY <<< "$TASK_IDS"

echo "PYTHON_BIN=$PYTHON_BIN"
echo "BASE_MODEL_PATH=$BASE_MODEL_PATH"
echo "VISION_TOWER_PATH=$VISION_TOWER_PATH"
echo "UCIT_ROOT=$UCIT_ROOT"
echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "TASK_IDS=${TASK_ID_ARRAY[*]}"
echo "SAMPLES_PER_TASK=$SAMPLES_PER_TASK"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LOG_FILE=$LOG_FILE"

"$PYTHON_BIN" scripts/MCITlib/Analysis/analyze_description_drift.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --vision-tower-path "$VISION_TOWER_PATH" \
    --ucit-root "$UCIT_ROOT" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --task-ids "${TASK_ID_ARRAY[@]}" \
    --samples-per-task "$SAMPLES_PER_TASK" \
    --device "$DEVICE" \
    --seed "$SEED"

echo "Medium outputs saved to: $OUTPUT_DIR"
