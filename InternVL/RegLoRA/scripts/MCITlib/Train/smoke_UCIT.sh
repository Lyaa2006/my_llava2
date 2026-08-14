#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

export HARD_PATH=${HARD_PATH:-/mnt/lyaa/MCITlib}
export DATA_SUFFIX=${DATA_SUFFIX:--smoke}
export DATA_CONFIG_DIR=${DATA_CONFIG_DIR:-$HARD_PATH/configs/data_configs/UCIT}
export TRAIN_CONFIG_ROOT=${TRAIN_CONFIG_ROOT:-$HARD_PATH/configs/train_configs/RegLoRA/InternVL/UCIT_smoke}
export EVAL_CONFIG_ROOT=${EVAL_CONFIG_ROOT:-$TRAIN_CONFIG_ROOT}
export START_TASK=${START_TASK:-1}
export END_TASK=${END_TASK:-6}

LOG_DIR=${LOG_DIR:-$HARD_PATH/logs/InternVL_RegLoRA_UCIT_smoke}
mkdir -p "$LOG_DIR"
LOG_FILE=${LOG_FILE:-$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log}

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Smoke log: $LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Tasks: $START_TASK-$END_TASK"

bash scripts/MCITlib/Train/train_UCIT.sh
