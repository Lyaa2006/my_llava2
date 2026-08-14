#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
HARD_PATH="${HARD_PATH:-$MCITLIB_ROOT}"

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export MASTER_PORT="${MASTER_PORT:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')}"

LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_RegLoRA_DCL_${DCL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "NCCL_IB_DISABLE: ${NCCL_IB_DISABLE}"
echo "NCCL_P2P_DISABLE: ${NCCL_P2P_DISABLE}"

MODEL_CONFIG="$HARD_PATH/configs/model_configs/internvl.json"

if [ "${DCL_SMOKE:-0}" = "1" ]; then
    DATA_TASK1="$HARD_PATH/configs/data_configs/MLLM-DCL/RS-smoke.json"
    DATA_TASK2="$HARD_PATH/configs/data_configs/MLLM-DCL/Med-smoke.json"
    DATA_TASK3="$HARD_PATH/configs/data_configs/MLLM-DCL/AD-smoke.json"
    DATA_TASK4="$HARD_PATH/configs/data_configs/MLLM-DCL/Sci-smoke.json"
    DATA_TASK5="$HARD_PATH/configs/data_configs/MLLM-DCL/Fin-smoke.json"
    TRAIN_TASK1="$HARD_PATH/configs/train_configs/RegLoRA/InternVL/MLLM-DCL/train/task1_smoke.json"
    TRAIN_TASK2="$HARD_PATH/configs/train_configs/RegLoRA/InternVL/MLLM-DCL/train/task2_smoke.json"
    TRAIN_TASK3="$HARD_PATH/configs/train_configs/RegLoRA/InternVL/MLLM-DCL/train/task3_smoke.json"
    TRAIN_TASK4="$HARD_PATH/configs/train_configs/RegLoRA/InternVL/MLLM-DCL/train/task4_smoke.json"
    TRAIN_TASK5="$HARD_PATH/configs/train_configs/RegLoRA/InternVL/MLLM-DCL/train/task5_smoke.json"
    DCL_SKIP_EVAL="${DCL_SKIP_EVAL:-1}"
else
    DATA_TASK1="$HARD_PATH/configs/data_configs/MLLM-DCL/RS.json"
    DATA_TASK2="$HARD_PATH/configs/data_configs/MLLM-DCL/Med.json"
    DATA_TASK3="$HARD_PATH/configs/data_configs/MLLM-DCL/AD.json"
    DATA_TASK4="$HARD_PATH/configs/data_configs/MLLM-DCL/Sci.json"
    DATA_TASK5="$HARD_PATH/configs/data_configs/MLLM-DCL/Fin.json"
    TRAIN_TASK1="$HARD_PATH/configs/train_configs/RegLoRA/InternVL/MLLM-DCL/train/task1.json"
    TRAIN_TASK2="$HARD_PATH/configs/train_configs/RegLoRA/InternVL/MLLM-DCL/train/task2.json"
    TRAIN_TASK3="$HARD_PATH/configs/train_configs/RegLoRA/InternVL/MLLM-DCL/train/task3.json"
    TRAIN_TASK4="$HARD_PATH/configs/train_configs/RegLoRA/InternVL/MLLM-DCL/train/task4.json"
    TRAIN_TASK5="$HARD_PATH/configs/train_configs/RegLoRA/InternVL/MLLM-DCL/train/task5.json"
    DCL_SKIP_EVAL="${DCL_SKIP_EVAL:-0}"
fi

bash scripts/MCITlib/Train/Task1.sh "$MODEL_CONFIG" "$DATA_TASK1" "$TRAIN_TASK1"
if [ "$DCL_SKIP_EVAL" != "1" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 1
fi

bash scripts/MCITlib/Train/Taskn.sh "$MODEL_CONFIG" "$DATA_TASK2" "$TRAIN_TASK2"
if [ "$DCL_SKIP_EVAL" != "1" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 2
fi

bash scripts/MCITlib/Train/Taskn.sh "$MODEL_CONFIG" "$DATA_TASK3" "$TRAIN_TASK3"
if [ "$DCL_SKIP_EVAL" != "1" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 3
fi

bash scripts/MCITlib/Train/Taskn.sh "$MODEL_CONFIG" "$DATA_TASK4" "$TRAIN_TASK4"
if [ "$DCL_SKIP_EVAL" != "1" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 4
fi

bash scripts/MCITlib/Train/Taskn.sh "$MODEL_CONFIG" "$DATA_TASK5" "$TRAIN_TASK5"
if [ "$DCL_SKIP_EVAL" != "1" ]; then
    bash scripts/MCITlib/Eval_MLLM_DCL/Eval_finetune1.sh 5
fi
