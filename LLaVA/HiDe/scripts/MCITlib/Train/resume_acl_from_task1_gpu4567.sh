#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIDE_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$HIDE_ROOT/../..")"

RUN_ROOT="${1:-$MCITLIB_ROOT/checkpoints/MLLM-ACL/LLaVA/HiDe/acl_full_20260722_075427}"
CFG_DIR="$RUN_ROOT/generated_configs"
MODEL_CFG="$MCITLIB_ROOT/configs/model_configs/llava.json"
DATA_DIR="$MCITLIB_ROOT/configs/data_configs/MLLM-ACL"

export HARD_PATH="${HARD_PATH:-$MCITLIB_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/lyaa_triton_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs/MLLM-ACL}"
mkdir -p "$LOG_DIR"

RUN_NAME="$(basename "$RUN_ROOT")"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_NAME}_resume_gpu4567_full.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Using RUN_ROOT=$RUN_ROOT"
echo "Global batch stays unchanged: 4 GPUs x per_device_train_batch_size=4 x grad_acc=1 = 16"

cd "$HIDE_ROOT"

run_eval_task2() {
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_OCR.sh "$MODEL_CFG" "$DATA_DIR/OCR.json" "$CFG_DIR/eval_task2.json"
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_Math.sh "$MODEL_CFG" "$DATA_DIR/Math.json" "$CFG_DIR/eval_task2.json"
}

run_eval_task3() {
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_OCR.sh "$MODEL_CFG" "$DATA_DIR/OCR.json" "$CFG_DIR/eval_task3.json"
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_Math.sh "$MODEL_CFG" "$DATA_DIR/Math.json" "$CFG_DIR/eval_task3.json"
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_VP.sh "$MODEL_CFG" "$DATA_DIR/VP.json" "$CFG_DIR/eval_task3.json"
}

run_eval_task4() {
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_OCR.sh "$MODEL_CFG" "$DATA_DIR/OCR.json" "$CFG_DIR/eval_task4.json"
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_Math.sh "$MODEL_CFG" "$DATA_DIR/Math.json" "$CFG_DIR/eval_task4.json"
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_VP.sh "$MODEL_CFG" "$DATA_DIR/VP.json" "$CFG_DIR/eval_task4.json"
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_APP.sh "$MODEL_CFG" "$DATA_DIR/APP.json" "$CFG_DIR/eval_task4.json"
}

echo "=== Task2 train ==="
bash scripts/MCITlib/Train/Taskn.sh "$MODEL_CFG" "$DATA_DIR/Math.json" "$CFG_DIR/train_task2.json"
echo "=== Task2 eval ==="
run_eval_task2

echo "=== Task3 train ==="
bash scripts/MCITlib/Train/Taskn.sh "$MODEL_CFG" "$DATA_DIR/VP.json" "$CFG_DIR/train_task3.json"
echo "=== Task3 eval ==="
run_eval_task3

echo "=== Task4 train ==="
bash scripts/MCITlib/Train/Taskn.sh "$MODEL_CFG" "$DATA_DIR/APP.json" "$CFG_DIR/train_task4.json"
echo "=== Task4 eval ==="
run_eval_task4
