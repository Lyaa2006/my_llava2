#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
HARD_PATH="${HARD_PATH:-$MCITLIB_ROOT}"

cd "$PROJECT_ROOT"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3,4,5,6}"
export GPU_NUM_OVERRIDE="${GPU_NUM_OVERRIDE:-4}"

LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs/MLLM-DCL}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/hide_mllm_dcl_smoke_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "GPU_NUM_OVERRIDE=$GPU_NUM_OVERRIDE"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"

export HIDE_DCL_SMOKE=1

bash scripts/MCITlib/Train/Task1.sh \
    "$HARD_PATH/configs/model_configs/llava.json" \
    "$HARD_PATH/configs/data_configs/MLLM-DCL/RS-smoke.json" \
    "$HARD_PATH/configs/train_configs/HiDe/LLaVA/MLLM-DCL/train/task1_smoke.json"
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 1

bash scripts/MCITlib/Train/Taskn.sh \
    "$HARD_PATH/configs/model_configs/llava.json" \
    "$HARD_PATH/configs/data_configs/MLLM-DCL/Med-smoke.json" \
    "$HARD_PATH/configs/train_configs/HiDe/LLaVA/MLLM-DCL/train/task2_smoke.json"
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 2

bash scripts/MCITlib/Train/Taskn.sh \
    "$HARD_PATH/configs/model_configs/llava.json" \
    "$HARD_PATH/configs/data_configs/MLLM-DCL/AD-smoke.json" \
    "$HARD_PATH/configs/train_configs/HiDe/LLaVA/MLLM-DCL/train/task3_smoke.json"
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 3

bash scripts/MCITlib/Train/Taskn.sh \
    "$HARD_PATH/configs/model_configs/llava.json" \
    "$HARD_PATH/configs/data_configs/MLLM-DCL/Sci-smoke.json" \
    "$HARD_PATH/configs/train_configs/HiDe/LLaVA/MLLM-DCL/train/task4_smoke.json"
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 4

bash scripts/MCITlib/Train/Taskn.sh \
    "$HARD_PATH/configs/model_configs/llava.json" \
    "$HARD_PATH/configs/data_configs/MLLM-DCL/Fin-smoke.json" \
    "$HARD_PATH/configs/train_configs/HiDe/LLaVA/MLLM-DCL/train/task5_smoke.json"
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 5
