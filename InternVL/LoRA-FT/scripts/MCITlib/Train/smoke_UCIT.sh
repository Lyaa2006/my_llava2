#!/bin/bash

set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
HARD_PATH=$(cd "$SCRIPT_ROOT/../.." && pwd)
LOG_ROOT=${MCIT_LOG_ROOT:-$HARD_PATH/logs/InternVL/LoRA-FT}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG="$LOG_ROOT/smoke_ucit_${TIMESTAMP}.log"

mkdir -p "$LOG_ROOT" "$HARD_PATH/results/UCIT/each_dataset_smoke"
cd "$SCRIPT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export MASTER_PORT="${MASTER_PORT:-9001}"

if ! command -v torchrun >/dev/null 2>&1 \
    && [ -x "/home/lyaa/miniconda3/envs/MCITlib/bin/torchrun" ]; then
    export PATH="/home/lyaa/miniconda3/envs/MCITlib/bin:$PATH"
fi

exec > >(tee -a "$RUN_LOG") 2>&1

echo "Smoke log: $RUN_LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "MASTER_PORT=$MASTER_PORT"
echo "torchrun=$(command -v torchrun || echo missing)"

MODEL_CONFIG="$HARD_PATH/configs/model_configs/internvl.json"
TRAIN_ROOT="$HARD_PATH/configs/train_configs/LoRA-FT/InternVL/UCIT/train"
EVAL_ROOT="$HARD_PATH/configs/train_configs/LoRA-FT/InternVL/UCIT/eval"
DATA_ROOT="$HARD_PATH/configs/data_configs/UCIT"

bash scripts/MCITlib/Train/Task1.sh \
    "$MODEL_CONFIG" "$DATA_ROOT/ImageNet-R-smoke.json" "$TRAIN_ROOT/task1_smoke.json"
bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" "$DATA_ROOT/ArxivQA-smoke.json" "$TRAIN_ROOT/task2_smoke.json"
bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" "$DATA_ROOT/VizWiz-smoke.json" "$TRAIN_ROOT/task3_smoke.json"
bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" "$DATA_ROOT/IconQA-smoke.json" "$TRAIN_ROOT/task4_smoke.json"
bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" "$DATA_ROOT/CLEVR-Math-smoke.json" "$TRAIN_ROOT/task5_smoke.json"
bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CONFIG" "$DATA_ROOT/Flickr30k-smoke.json" "$TRAIN_ROOT/task6_smoke.json"

if [ "${MCIT_SKIP_EVAL:-0}" != "1" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh \
        "$MODEL_CONFIG" "$DATA_ROOT/ImageNet-R-smoke.json" "$EVAL_ROOT/task6_smoke.json"
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh \
        "$MODEL_CONFIG" "$DATA_ROOT/ArxivQA-smoke.json" "$EVAL_ROOT/task6_smoke.json"
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh \
        "$MODEL_CONFIG" "$DATA_ROOT/VizWiz-smoke.json" "$EVAL_ROOT/task6_smoke.json"
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh \
        "$MODEL_CONFIG" "$DATA_ROOT/IconQA-smoke.json" "$EVAL_ROOT/task6_smoke.json"
    bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh \
        "$MODEL_CONFIG" "$DATA_ROOT/CLEVR-Math-smoke.json" "$EVAL_ROOT/task6_smoke.json"
    bash scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh \
        "$MODEL_CONFIG" "$DATA_ROOT/Flickr30k-smoke.json" "$EVAL_ROOT/task6_smoke.json"
fi

echo "LoRA-FT InternVL UCIT smoke completed."
