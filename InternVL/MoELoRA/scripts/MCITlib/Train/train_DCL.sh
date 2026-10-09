#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$PROJECT_ROOT"

MODE="${1:-full}"
case "$MODE" in
    full|smoke) ;;
    *)
        echo "Usage: $0 [full|smoke]" >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
export PYTHON_BIN="${PYTHON_BIN:-__EXTERNAL_ROOT__/bin/python}"
export TORCHRUN_BIN="${TORCHRUN_BIN:-__EXTERNAL_ROOT__/bin/torchrun}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [[ "$MODE" == "smoke" ]]; then
    DATA_SUFFIX="-smoke"
    TRAIN_SUFFIX="_smoke"
else
    DATA_SUFFIX=""
    TRAIN_SUFFIX=""
fi

MODEL_CONFIG="$REPO_ROOT/configs/model_configs/internvl.json"
TRAIN_CONFIG_ROOT="$REPO_ROOT/configs/train_configs/MoELoRA/InternVL/MLLM-DCL/train"
EVAL_CONFIG_ROOT="$REPO_ROOT/configs/train_configs/MoELoRA/InternVL/MLLM-DCL/eval"
DATA_CONFIG_ROOT="$REPO_ROOT/configs/data_configs/MLLM-DCL"

bash scripts/MCITlib/Train/Task1.sh \
    "$MODEL_CONFIG" \
    "$DATA_CONFIG_ROOT/RS${DATA_SUFFIX}.json" \
    "$TRAIN_CONFIG_ROOT/task1${TRAIN_SUFFIX}.json"
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 1 "$MODE"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" \
    "$DATA_CONFIG_ROOT/Med${DATA_SUFFIX}.json" \
    "$TRAIN_CONFIG_ROOT/task2${TRAIN_SUFFIX}.json"
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 2 "$MODE"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" \
    "$DATA_CONFIG_ROOT/AD${DATA_SUFFIX}.json" \
    "$TRAIN_CONFIG_ROOT/task3${TRAIN_SUFFIX}.json"
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 3 "$MODE"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" \
    "$DATA_CONFIG_ROOT/Sci${DATA_SUFFIX}.json" \
    "$TRAIN_CONFIG_ROOT/task4${TRAIN_SUFFIX}.json"
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 4 "$MODE"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" \
    "$DATA_CONFIG_ROOT/Fin${DATA_SUFFIX}.json" \
    "$TRAIN_CONFIG_ROOT/task5${TRAIN_SUFFIX}.json"
bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 5 "$MODE"
