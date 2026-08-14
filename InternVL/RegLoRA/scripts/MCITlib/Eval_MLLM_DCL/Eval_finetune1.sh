#!/bin/bash
set -e

TASK_ID=$1
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
HARD_PATH="${HARD_PATH:-$MCITLIB_ROOT}"

cd "$PROJECT_ROOT"

LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/eval_RegLoRA_DCL_task${TASK_ID}_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

if [ "${DCL_SMOKE:-0}" = "1" ]; then
    EVAL_ROOT="$HARD_PATH/configs/train_configs/RegLoRA/InternVL/MLLM-DCL/eval"
    EVAL_TASK1="$EVAL_ROOT/task1_smoke.json"
    EVAL_TASK2="$EVAL_ROOT/task2_smoke.json"
    EVAL_TASK3="$EVAL_ROOT/task3_smoke.json"
    EVAL_TASK4="$EVAL_ROOT/task4_smoke.json"
    EVAL_TASK5="$EVAL_ROOT/task5_smoke.json"
else
    EVAL_ROOT="$HARD_PATH/configs/train_configs/RegLoRA/InternVL/MLLM-DCL/eval"
    EVAL_TASK1="$EVAL_ROOT/task1.json"
    EVAL_TASK2="$EVAL_ROOT/task2.json"
    EVAL_TASK3="$EVAL_ROOT/task3.json"
    EVAL_TASK4="$EVAL_ROOT/task4.json"
    EVAL_TASK5="$EVAL_ROOT/task5.json"
fi

if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $EVAL_TASK1
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $EVAL_TASK2
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $EVAL_TASK2
elif [ "$TASK_ID" == "3" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $EVAL_TASK3
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $EVAL_TASK3
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $EVAL_TASK3
elif [ "$TASK_ID" == "4" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $EVAL_TASK4
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $EVAL_TASK4
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $EVAL_TASK4
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json $EVAL_TASK4
else
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $EVAL_TASK5
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $EVAL_TASK5
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $EVAL_TASK5
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json $EVAL_TASK5
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_fin.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/Fin.json $EVAL_TASK5
fi
