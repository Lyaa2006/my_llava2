#!/bin/bash
set -e

TASK_ID=$1
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
HARD_PATH="${HARD_PATH:-$MCITLIB_ROOT}"

if [ "${HIDE_ACL_SMOKE:-0}" = "1" ]; then
    DATA_DIR="$HARD_PATH/configs/data_configs/MLLM-ACL"
    EVAL_DIR="$HARD_PATH/configs/train_configs/HiDe/LLaVA/MLLM-ACL/eval"
    DATA_OCR="$DATA_DIR/OCR-smoke.json"
    DATA_MATH="$DATA_DIR/Math-smoke.json"
    DATA_VP="$DATA_DIR/VP-smoke.json"
    DATA_APP="$DATA_DIR/APP-smoke.json"
    EVAL_TASK1="$EVAL_DIR/task1_smoke.json"
    EVAL_TASK2="$EVAL_DIR/task2_smoke.json"
    EVAL_TASK3="$EVAL_DIR/task3_smoke.json"
    EVAL_TASK4="$EVAL_DIR/task4_smoke.json"
else
    DATA_DIR="$HARD_PATH/configs/data_configs/MLLM-ACL"
    EVAL_DIR="$HARD_PATH/configs/train_configs/HiDe/LLaVA/MLLM-ACL/eval"
    DATA_OCR="$DATA_DIR/OCR.json"
    DATA_MATH="$DATA_DIR/Math.json"
    DATA_VP="$DATA_DIR/VP.json"
    DATA_APP="$DATA_DIR/APP.json"
    EVAL_TASK1="$EVAL_DIR/task1.json"
    EVAL_TASK2="$EVAL_DIR/task2.json"
    EVAL_TASK3="$EVAL_DIR/task3.json"
    EVAL_TASK4="$EVAL_DIR/task4.json"
fi

if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_OCR.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_OCR" "$EVAL_TASK1"
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_OCR.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_OCR" "$EVAL_TASK2"
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_Math.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_MATH" "$EVAL_TASK2"
elif [ "$TASK_ID" == "3" ]; then
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_OCR.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_OCR" "$EVAL_TASK3"
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_Math.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_MATH" "$EVAL_TASK3"
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_VP.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_VP" "$EVAL_TASK3"
else
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_OCR.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_OCR" "$EVAL_TASK4"
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_Math.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_MATH" "$EVAL_TASK4"
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_VP.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_VP" "$EVAL_TASK4"
    bash scripts/MCITlib/Eval_MLLM_ACL/eval_APP.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_APP" "$EVAL_TASK4"
fi
