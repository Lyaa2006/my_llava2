#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
HARD_PATH="${HARD_PATH:-$MCITLIB_ROOT}"

LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs/MLLM-ACL}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_ACL_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"

bash scripts/MCITlib/Train/Task1.sh \
    "$HARD_PATH/configs/model_configs/llava.json" \
    "$HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json" \
    "$HARD_PATH/configs/train_configs/HiDe/LLaVA/MLLM-ACL/train/task1.json"
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 1

bash scripts/MCITlib/Train/Taskn.sh \
    "$HARD_PATH/configs/model_configs/llava.json" \
    "$HARD_PATH/configs/data_configs/MLLM-ACL/Math.json" \
    "$HARD_PATH/configs/train_configs/HiDe/LLaVA/MLLM-ACL/train/task2.json"
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 2

bash scripts/MCITlib/Train/Taskn.sh \
    "$HARD_PATH/configs/model_configs/llava.json" \
    "$HARD_PATH/configs/data_configs/MLLM-ACL/VP.json" \
    "$HARD_PATH/configs/train_configs/HiDe/LLaVA/MLLM-ACL/train/task3.json"
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 3

bash scripts/MCITlib/Train/Taskn.sh \
    "$HARD_PATH/configs/model_configs/llava.json" \
    "$HARD_PATH/configs/data_configs/MLLM-ACL/APP.json" \
    "$HARD_PATH/configs/train_configs/HiDe/LLaVA/MLLM-ACL/train/task4.json"
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 4
