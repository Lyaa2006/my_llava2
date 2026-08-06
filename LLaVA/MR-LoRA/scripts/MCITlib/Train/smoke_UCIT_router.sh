#!/bin/bash

set -euo pipefail

HARD_PATH=${HARD_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-cache/MR-LoRA}"
export TRITON_CACHE_PATH="$TRITON_CACHE_DIR"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p "$TRITON_CACHE_DIR"

MODEL_CONFIG="$HARD_PATH/configs/model_configs/llava.json"
TRAIN_ROOT="$HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train_router"
EVAL_ROOT="$HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/eval_router"

bash scripts/MCITlib/Train/Task1_router.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json" \
    "$TRAIN_ROOT/task1_smoke.json"

bash scripts/MCITlib/Train/Task1_router.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/ArxivQA.json" \
    "$TRAIN_ROOT/task2_smoke.json"

bash scripts/MCITlib/Train/Task1_router.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/VizWiz.json" \
    "$TRAIN_ROOT/task3_smoke.json"

bash scripts/MCITlib/Train/Task1_router.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/IconQA.json" \
    "$TRAIN_ROOT/task4_smoke.json"

bash scripts/MCITlib/Train/Task1_router.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json" \
    "$TRAIN_ROOT/task5_smoke.json"

bash scripts/MCITlib/Train/Task1_router.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/Flickr30k.json" \
    "$TRAIN_ROOT/task6_smoke.json"

bash scripts/MCITlib/Eval_UCIT_router/eval_imagenet.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json" \
    "$EVAL_ROOT/task6_smoke.json"

bash scripts/MCITlib/Eval_UCIT_router/eval_arxivqa.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/ArxivQA.json" \
    "$EVAL_ROOT/task6_smoke.json"

bash scripts/MCITlib/Eval_UCIT_router/eval_vizwiz.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/VizWiz.json" \
    "$EVAL_ROOT/task6_smoke.json"

bash scripts/MCITlib/Eval_UCIT_router/eval_iconqa.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/IconQA.json" \
    "$EVAL_ROOT/task6_smoke.json"

bash scripts/MCITlib/Eval_UCIT_router/eval_clevr.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json" \
    "$EVAL_ROOT/task6_smoke.json"

bash scripts/MCITlib/Eval_UCIT_router/eval_flickr30k.sh \
    "$MODEL_CONFIG" \
    "$HARD_PATH/configs/data_configs/UCIT/Flickr30k.json" \
    "$EVAL_ROOT/task6_smoke.json"

MCIT_RESULTS_UCIT="$HARD_PATH/results/UCIT/each_dataset_smoke" \
    python3 scripts/UCIT_extract_router_result.py
