#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_path_resolver.sh"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
HARD_PATH="${HARD_PATH:-$MCITLIB_ROOT}"
if [ -d "$HARD_PATH/configs" ]; then
    CONFIG_ROOT="$HARD_PATH/configs"
else
    CONFIG_ROOT="$HARD_PATH"
fi

cd "$PROJECT_ROOT"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export MASTER_PORT="${MASTER_PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')}"

LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_UCIT_${UCIT_RUN_ID:-$(date +%Y%m%d_%H%M%S)}.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"

if [ "${UCIT_SMOKE:-0}" = "1" ]; then
    UCIT_TASKS="${UCIT_TASKS:-1,2,3,4,5,6}"
    UCIT_EVAL_MODE="${UCIT_EVAL_MODE:-per_task}"
    export UCIT_SMOKE_EVAL_LIMIT="${UCIT_SMOKE_EVAL_LIMIT:-32}"
    UCIT_SKIP_VIZWIZ=0
    UCIT_SKIP_FLICKR=0
    DATA_TASK1="$CONFIG_ROOT/data_configs/UCIT/ImageNet-R-smoke.json"
    DATA_TASK2="$CONFIG_ROOT/data_configs/UCIT/ArxivQA-smoke.json"
    DATA_TASK3="$CONFIG_ROOT/data_configs/UCIT/VizWiz-smoke.json"
    DATA_TASK4="$CONFIG_ROOT/data_configs/UCIT/IconQA-smoke.json"
    DATA_TASK5="$CONFIG_ROOT/data_configs/UCIT/CLEVR-Math-smoke.json"
    DATA_TASK6="$CONFIG_ROOT/data_configs/UCIT/Flickr30k-smoke.json"
    TRAIN_TASK1="$CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/train/task1_smoke.json"
    TRAIN_TASK2="$CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/train/task2_smoke.json"
    TRAIN_TASK3="$CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/train/task3_smoke.json"
    TRAIN_TASK4="$CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/train/task4_smoke.json"
    TRAIN_TASK5="$CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/train/task5_smoke.json"
    TRAIN_TASK6="$CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/train/task6_smoke.json"
else
    UCIT_TASKS="${UCIT_TASKS:-1,2,3,4,5,6}"
    UCIT_EVAL_MODE="${UCIT_EVAL_MODE:-per_task}"
    DATA_TASK1="$CONFIG_ROOT/data_configs/UCIT/ImageNet-R.json"
    DATA_TASK2="$CONFIG_ROOT/data_configs/UCIT/ArxivQA.json"
    DATA_TASK3="$CONFIG_ROOT/data_configs/UCIT/VizWiz.json"
    DATA_TASK4="$CONFIG_ROOT/data_configs/UCIT/IconQA.json"
    DATA_TASK5="$CONFIG_ROOT/data_configs/UCIT/CLEVR-Math.json"
    DATA_TASK6="$CONFIG_ROOT/data_configs/UCIT/Flickr30k.json"
    TRAIN_TASK1="$CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/train/task1.json"
    TRAIN_TASK2="$CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/train/task2.json"
    TRAIN_TASK3="$CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/train/task3.json"
    TRAIN_TASK4="$CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/train/task4.json"
    TRAIN_TASK5="$CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/train/task5.json"
    TRAIN_TASK6="$CONFIG_ROOT/train_configs/HiDESC/LLaVA/UCIT/train/task6.json"
fi
export UCIT_TASKS UCIT_EVAL_MODE
export UCIT_SKIP_VIZWIZ="${UCIT_SKIP_VIZWIZ:-0}"
export UCIT_SKIP_FLICKR="${UCIT_SKIP_FLICKR:-0}"

bash scripts/MCITlib/Train/Task1.sh \
    $CONFIG_ROOT/model_configs/llava.json \
    $DATA_TASK1 \
    $TRAIN_TASK1
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 1
bash scripts/MCITlib/Train/Taskn.sh \
    $CONFIG_ROOT/model_configs/llava.json \
    $DATA_TASK2 \
    $TRAIN_TASK2
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 2
bash scripts/MCITlib/Train/Taskn.sh \
    $CONFIG_ROOT/model_configs/llava.json \
    $DATA_TASK3 \
    $TRAIN_TASK3
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 3

bash scripts/MCITlib/Train/Taskn.sh \
    $CONFIG_ROOT/model_configs/llava.json \
    $DATA_TASK4 \
    $TRAIN_TASK4
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 4

bash scripts/MCITlib/Train/Taskn.sh \
    $CONFIG_ROOT/model_configs/llava.json \
    $DATA_TASK5 \
    $TRAIN_TASK5
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 5

bash scripts/MCITlib/Train/Taskn.sh \
    $CONFIG_ROOT/model_configs/llava.json \
    $DATA_TASK6 \
    $TRAIN_TASK6
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 6
