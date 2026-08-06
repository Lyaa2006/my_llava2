#!/bin/bash

set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
HARD_PATH=${HARD_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}
LOG_DIR=${LOG_DIR:-"$HARD_PATH/logs/MLLM-DCL/InternVL/OLoRA/full"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG=${RUN_LOG:-"$LOG_DIR/olora_internvl_mllm_dcl_full_${TIMESTAMP}.log"}

mkdir -p "$LOG_DIR"
cd "$SCRIPT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export PYTHONUNBUFFERED=1

exec > >(tee -a "$RUN_LOG") 2>&1

echo "Full run log: $RUN_LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "Global batch per task = gpu_num(4) * batch_size(4) * grad_acc(1) = 16"
echo "Original global batch = gpu_num(2) * batch_size(4) * grad_acc(2) = 16"

bash scripts/MCITlib/Train/train_DCL.sh
