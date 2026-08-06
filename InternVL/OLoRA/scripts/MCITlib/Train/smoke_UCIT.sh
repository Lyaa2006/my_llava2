#!/bin/bash

set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
HARD_PATH=${HARD_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}
LOG_DIR="$HARD_PATH/logs/o-lora"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG="$LOG_DIR/smoke_ucit_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"
cd "$SCRIPT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,7}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "Smoke log: $RUN_LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"

MODEL_CONFIG="$HARD_PATH/configs/model_configs/internvl.json"
TRAIN_ROOT="$HARD_PATH/configs/train_configs/OLoRA/InternVL/UCIT/train"
EVAL_ROOT="$HARD_PATH/configs/train_configs/OLoRA/InternVL/UCIT/eval"


bash scripts/MCITlib/Train/Task1.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/ImageNet-R-smoke.json" \
    "$TRAIN_ROOT/task1_smoke.json"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/ArxivQA-smoke.json" \
    "$TRAIN_ROOT/task2_smoke.json"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/VizWiz-smoke.json" \
    "$TRAIN_ROOT/task3_smoke.json"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/IconQA-smoke.json" \
    "$TRAIN_ROOT/task4_smoke.json"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/CLEVR-Math-smoke.json" \
    "$TRAIN_ROOT/task5_smoke.json"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/Flickr30k-smoke.json" \
    "$TRAIN_ROOT/task6_smoke.json"

bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/ImageNet-R-smoke.json" \
    "$EVAL_ROOT/task6_smoke.json"

bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/ArxivQA-smoke.json" \
    "$EVAL_ROOT/task6_smoke.json"

bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/VizWiz-smoke.json" \
    "$EVAL_ROOT/task6_smoke.json"

bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/IconQA-smoke.json" \
    "$EVAL_ROOT/task6_smoke.json"

bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/CLEVR-Math-smoke.json" \
    "$EVAL_ROOT/task6_smoke.json"

bash scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/Flickr30k-smoke.json" \
    "$EVAL_ROOT/task6_smoke.json"
