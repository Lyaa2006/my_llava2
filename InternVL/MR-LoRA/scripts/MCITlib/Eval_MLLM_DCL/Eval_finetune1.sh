#!/bin/bash
set -e

TASK_ID=$1
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
HARD_PATH="${HARD_PATH:-$MCITLIB_ROOT}"

cd "$PROJECT_ROOT"

if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/eval/task1.json
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/eval/task2.json
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/eval/task2.json
elif [ "$TASK_ID" == "3" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/eval/task3.json
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/eval/task3.json
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/eval/task3.json
elif [ "$TASK_ID" == "4" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/eval/task4.json
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/eval/task4.json
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/eval/task4.json
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/eval/task4.json
else
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/AD.json $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/eval/task5.json
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/RS.json $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/eval/task5.json
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/Med.json $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/eval/task5.json
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/eval/task5.json
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_fin.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/MLLM-DCL/Fin.json $HARD_PATH/configs/train_configs/MR-LoRA/InternVL/MLLM-DCL/eval/task5.json
fi
