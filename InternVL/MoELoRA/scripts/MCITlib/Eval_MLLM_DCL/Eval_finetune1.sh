#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$PROJECT_ROOT"

TASK_ID=$1
MODE="${2:-full}"

case "$MODE" in
    full|smoke) ;;
    *)
        echo "Usage: $0 <task_id> [full|smoke]" >&2
        exit 2
        ;;
esac

if [[ "$MODE" == "smoke" ]]; then
    DATA_SUFFIX="-smoke"
    TRAIN_SUFFIX="_smoke"
else
    DATA_SUFFIX=""
    TRAIN_SUFFIX=""
fi

MODEL_CONFIG="$REPO_ROOT/configs/model_configs/internvl.json"
EVAL_CONFIG_ROOT="$REPO_ROOT/configs/train_configs/MoELoRA/InternVL/MLLM-DCL/eval"
DATA_CONFIG_ROOT="$REPO_ROOT/configs/data_configs/MLLM-DCL"

if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh "$MODEL_CONFIG" "$DATA_CONFIG_ROOT/RS${DATA_SUFFIX}.json" "$EVAL_CONFIG_ROOT/task1${TRAIN_SUFFIX}.json"
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh "$MODEL_CONFIG" "$DATA_CONFIG_ROOT/Med${DATA_SUFFIX}.json" "$EVAL_CONFIG_ROOT/task2${TRAIN_SUFFIX}.json"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh "$MODEL_CONFIG" "$DATA_CONFIG_ROOT/RS${DATA_SUFFIX}.json" "$EVAL_CONFIG_ROOT/task2${TRAIN_SUFFIX}.json"
elif [ "$TASK_ID" == "3" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh "$MODEL_CONFIG" "$DATA_CONFIG_ROOT/Med${DATA_SUFFIX}.json" "$EVAL_CONFIG_ROOT/task3${TRAIN_SUFFIX}.json"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh "$MODEL_CONFIG" "$DATA_CONFIG_ROOT/RS${DATA_SUFFIX}.json" "$EVAL_CONFIG_ROOT/task3${TRAIN_SUFFIX}.json"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh "$MODEL_CONFIG" "$DATA_CONFIG_ROOT/AD${DATA_SUFFIX}.json" "$EVAL_CONFIG_ROOT/task3${TRAIN_SUFFIX}.json"
elif [ "$TASK_ID" == "4" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh "$MODEL_CONFIG" "$DATA_CONFIG_ROOT/AD${DATA_SUFFIX}.json" "$EVAL_CONFIG_ROOT/task4${TRAIN_SUFFIX}.json"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh "$MODEL_CONFIG" "$DATA_CONFIG_ROOT/RS${DATA_SUFFIX}.json" "$EVAL_CONFIG_ROOT/task4${TRAIN_SUFFIX}.json"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh "$MODEL_CONFIG" "$DATA_CONFIG_ROOT/Med${DATA_SUFFIX}.json" "$EVAL_CONFIG_ROOT/task4${TRAIN_SUFFIX}.json"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh "$MODEL_CONFIG" "$DATA_CONFIG_ROOT/Sci${DATA_SUFFIX}.json" "$EVAL_CONFIG_ROOT/task4${TRAIN_SUFFIX}.json"
else
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh "$MODEL_CONFIG" "$DATA_CONFIG_ROOT/AD${DATA_SUFFIX}.json" "$EVAL_CONFIG_ROOT/task5${TRAIN_SUFFIX}.json"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh "$MODEL_CONFIG" "$DATA_CONFIG_ROOT/RS${DATA_SUFFIX}.json" "$EVAL_CONFIG_ROOT/task5${TRAIN_SUFFIX}.json"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh "$MODEL_CONFIG" "$DATA_CONFIG_ROOT/Med${DATA_SUFFIX}.json" "$EVAL_CONFIG_ROOT/task5${TRAIN_SUFFIX}.json"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh "$MODEL_CONFIG" "$DATA_CONFIG_ROOT/Sci${DATA_SUFFIX}.json" "$EVAL_CONFIG_ROOT/task5${TRAIN_SUFFIX}.json"
    bash scripts/MCITlib/Eval_MLLM_DCL/eval_fin.sh "$MODEL_CONFIG" "$DATA_CONFIG_ROOT/Fin${DATA_SUFFIX}.json" "$EVAL_CONFIG_ROOT/task5${TRAIN_SUFFIX}.json"
fi
