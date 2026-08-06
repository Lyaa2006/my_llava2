#!/bin/bash
set -e

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

RUN_MODE="full"
if [ "${UCIT_SMOKE:-0}" = "1" ]; then
    RUN_MODE="smoke"
fi

LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs/DISCO/UCIT}"
mkdir -p "$LOG_DIR"
LOG_STAMP="${UCIT_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_UCIT_${RUN_MODE}_${LOG_STAMP}.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"
echo "Run mode: $RUN_MODE"
echo "Project root: $PROJECT_ROOT"
echo "Config root: $CONFIG_ROOT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "NCCL_IB_DISABLE: ${NCCL_IB_DISABLE:-unset}"
echo "NCCL_P2P_DISABLE: ${NCCL_P2P_DISABLE:-unset}"
export PATH="/home/lyaa/miniconda3/envs/MCITlib_copy/bin:$PATH"
echo "Python: $(command -v python3)"
echo "Deepspeed: $(command -v deepspeed)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
echo "HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
echo "TRANSFORMERS_OFFLINE: $TRANSFORMERS_OFFLINE"
echo "HF_DATASETS_OFFLINE: $HF_DATASETS_OFFLINE"
if [ -z "${MASTER_PORT:-}" ]; then
    export MASTER_PORT="$((20000 + (${RANDOM:-12345} % 20000)))"
else
    export MASTER_PORT
fi
echo "MASTER_PORT: $MASTER_PORT"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$MCITLIB_ROOT/.triton/disco_ucit}"
mkdir -p "$TRITON_CACHE_DIR"
echo "TRITON_CACHE_DIR: $TRITON_CACHE_DIR"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S %Z')"

run_task() {
    local task_id="$1"
    local train_script="$2"
    local data_cfg="$3"
    local train_cfg="$4"

    bash "scripts/MCITlib/Train/$train_script" \
        "$CONFIG_ROOT/model_configs/llava.json" \
        "$data_cfg" \
        "$train_cfg"
    bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh "$task_id"
}

if [ "${UCIT_SMOKE:-0}" = "1" ]; then
    DATA_TASK1="$CONFIG_ROOT/data_configs/UCIT/ImageNet-R-smoke.json"
    DATA_TASK2="$CONFIG_ROOT/data_configs/UCIT/ArxivQA-smoke.json"
    DATA_TASK3="$CONFIG_ROOT/data_configs/UCIT/VizWiz-smoke.json"
    DATA_TASK4="$CONFIG_ROOT/data_configs/UCIT/IconQA-smoke.json"
    DATA_TASK5="$CONFIG_ROOT/data_configs/UCIT/CLEVR-Math-smoke.json"
    DATA_TASK6="$CONFIG_ROOT/data_configs/UCIT/Flickr30k-smoke.json"
    TRAIN_TASK1="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/train/task1_smoke.json"
    TRAIN_TASK2="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/train/task2_smoke.json"
    TRAIN_TASK3="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/train/task3_smoke.json"
    TRAIN_TASK4="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/train/task4_smoke.json"
    TRAIN_TASK5="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/train/task5_smoke.json"
    TRAIN_TASK6="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/train/task6_smoke.json"
else
    DATA_TASK1="$CONFIG_ROOT/data_configs/UCIT/ImageNet-R.json"
    DATA_TASK2="$CONFIG_ROOT/data_configs/UCIT/ArxivQA.json"
    DATA_TASK3="$CONFIG_ROOT/data_configs/UCIT/VizWiz.json"
    DATA_TASK4="$CONFIG_ROOT/data_configs/UCIT/IconQA.json"
    DATA_TASK5="$CONFIG_ROOT/data_configs/UCIT/CLEVR-Math.json"
    DATA_TASK6="$CONFIG_ROOT/data_configs/UCIT/Flickr30k.json"
    TRAIN_TASK1="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/train/task1.json"
    TRAIN_TASK2="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/train/task2.json"
    TRAIN_TASK3="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/train/task3.json"
    TRAIN_TASK4="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/train/task4.json"
    TRAIN_TASK5="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/train/task5.json"
    TRAIN_TASK6="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/train/task6.json"
fi

run_task 1 Task1.sh "$DATA_TASK1" "$TRAIN_TASK1"
run_task 2 Taskn.sh "$DATA_TASK2" "$TRAIN_TASK2"
run_task 3 Taskn.sh "$DATA_TASK3" "$TRAIN_TASK3"
run_task 4 Taskn.sh "$DATA_TASK4" "$TRAIN_TASK4"
run_task 5 Taskn.sh "$DATA_TASK5" "$TRAIN_TASK5"
run_task 6 Taskn.sh "$DATA_TASK6" "$TRAIN_TASK6"
