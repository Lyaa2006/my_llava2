#!/bin/bash

set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
HARD_PATH=${HARD_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}
LOG_DIR=${LOG_DIR:-"$HARD_PATH/logs/MLLM-DCL/InternVL/OLoRA"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG=${RUN_LOG:-"$LOG_DIR/olora_internvl_mllm_dcl_smoke_${TIMESTAMP}.log"}

mkdir -p "$LOG_DIR"
cd "$SCRIPT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export PYTHONUNBUFFERED=1

exec > >(tee -a "$RUN_LOG") 2>&1

echo "Smoke log: $RUN_LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"

MODEL_CONFIG="$HARD_PATH/configs/model_configs/internvl.json"
TRAIN_ROOT="$HARD_PATH/configs/train_configs/OLoRA/InternVL/MLLM-DCL/train"
EVAL_CONFIG="$HARD_PATH/configs/train_configs/OLoRA/InternVL/MLLM-DCL/eval/task5_smoke.json"

bash scripts/MCITlib/Train/Task1.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/MLLM-DCL/RS-smoke.json" \
    "$TRAIN_ROOT/task1_smoke.json"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/MLLM-DCL/Med-smoke.json" \
    "$TRAIN_ROOT/task2_smoke.json"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/MLLM-DCL/AD-smoke.json" \
    "$TRAIN_ROOT/task3_smoke.json"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/MLLM-DCL/Sci-smoke.json" \
    "$TRAIN_ROOT/task4_smoke.json"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/MLLM-DCL/Fin-smoke.json" \
    "$TRAIN_ROOT/task5_smoke.json"

bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/MLLM-DCL/RS-smoke.json" \
    "$EVAL_CONFIG"

bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/MLLM-DCL/Med-smoke.json" \
    "$EVAL_CONFIG"

bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/MLLM-DCL/AD-smoke.json" \
    "$EVAL_CONFIG"

bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/MLLM-DCL/Sci-smoke.json" \
    "$EVAL_CONFIG"

bash scripts/MCITlib/Eval_MLLM_DCL/eval_fin.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/MLLM-DCL/Fin-smoke.json" \
    "$EVAL_CONFIG"
