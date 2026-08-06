#!/bin/bash

set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
HARD_PATH=$(cd "$SCRIPT_ROOT/../.." && pwd)
cd "$SCRIPT_ROOT"

DATA_SUFFIX=""
TRAIN_SUFFIX=""
if [ "${MCIT_USE_SMOKE:-0}" = "1" ]; then
    DATA_SUFFIX="-smoke"
    TRAIN_SUFFIX="_smoke"
fi

bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/ImageNet-R${DATA_SUFFIX}.json \
    $HARD_PATH/configs/train_configs/LoRA-FT/InternVL/UCIT/train/task1${TRAIN_SUFFIX}.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 1

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/ArxivQA${DATA_SUFFIX}.json \
    $HARD_PATH/configs/train_configs/LoRA-FT/InternVL/UCIT/train/task2${TRAIN_SUFFIX}.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 2

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/VizWiz${DATA_SUFFIX}.json \
    $HARD_PATH/configs/train_configs/LoRA-FT/InternVL/UCIT/train/task3${TRAIN_SUFFIX}.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 3

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/IconQA${DATA_SUFFIX}.json \
    $HARD_PATH/configs/train_configs/LoRA-FT/InternVL/UCIT/train/task4${TRAIN_SUFFIX}.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 4

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math${DATA_SUFFIX}.json \
    $HARD_PATH/configs/train_configs/LoRA-FT/InternVL/UCIT/train/task5${TRAIN_SUFFIX}.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 5

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/Flickr30k${DATA_SUFFIX}.json \
    $HARD_PATH/configs/train_configs/LoRA-FT/InternVL/UCIT/train/task6${TRAIN_SUFFIX}.json
bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh 6
