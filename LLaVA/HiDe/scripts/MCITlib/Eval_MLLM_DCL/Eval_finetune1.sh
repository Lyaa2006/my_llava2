#!/bin/bash
set -e

TASK_ID=$1
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
HARD_PATH="${HARD_PATH:-$MCITLIB_ROOT}"

if [ "${HIDE_DCL_SMOKE:-0}" = "1" ]; then
    DATA_DIR="$HARD_PATH/configs/data_configs/MLLM-DCL"
    EVAL_DIR="$HARD_PATH/configs/train_configs/HiDe/LLaVA/MLLM-DCL/eval"
    DATA_RS="$DATA_DIR/RS-smoke.json"
    DATA_MED="$DATA_DIR/Med-smoke.json"
    DATA_AD="$DATA_DIR/AD-smoke.json"
    DATA_SCI="$DATA_DIR/Sci-smoke.json"
    DATA_FIN="$DATA_DIR/Fin-smoke.json"
    EVAL_TASK1="$EVAL_DIR/task1_smoke.json"
    EVAL_TASK2="$EVAL_DIR/task2_smoke.json"
    EVAL_TASK3="$EVAL_DIR/task3_smoke.json"
    EVAL_TASK4="$EVAL_DIR/task4_smoke.json"
    EVAL_TASK5="$EVAL_DIR/task5_smoke.json"
else
    DATA_DIR="$HARD_PATH/configs/data_configs/MLLM-DCL"
    EVAL_DIR="$HARD_PATH/configs/train_configs/HiDe/LLaVA/MLLM-DCL/eval"
    DATA_RS="$DATA_DIR/RS.json"
    DATA_MED="$DATA_DIR/Med.json"
    DATA_AD="$DATA_DIR/AD.json"
    DATA_SCI="$DATA_DIR/Sci.json"
    DATA_FIN="$DATA_DIR/Fin.json"
    EVAL_TASK1="$EVAL_DIR/task1.json"
    EVAL_TASK2="$EVAL_DIR/task2.json"
    EVAL_TASK3="$EVAL_DIR/task3.json"
    EVAL_TASK4="$EVAL_DIR/task4.json"
    EVAL_TASK5="$EVAL_DIR/task5.json"
fi

if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_RS" "$EVAL_TASK1"
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_MED" "$EVAL_TASK2"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_RS" "$EVAL_TASK2"
elif [ "$TASK_ID" == "3" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_MED" "$EVAL_TASK3"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_RS" "$EVAL_TASK3"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_AD" "$EVAL_TASK3"
elif [ "$TASK_ID" == "4" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_AD" "$EVAL_TASK4"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_RS" "$EVAL_TASK4"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_MED" "$EVAL_TASK4"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_SCI" "$EVAL_TASK4"
else
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_AD" "$EVAL_TASK5"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_RS" "$EVAL_TASK5"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_MED" "$EVAL_TASK5"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_SCI" "$EVAL_TASK5"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_fin.sh "$HARD_PATH/configs/model_configs/llava.json" "$DATA_FIN" "$EVAL_TASK5"
fi
