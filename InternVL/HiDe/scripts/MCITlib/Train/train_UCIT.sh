#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARD_PATH=$(cd "${SCRIPT_DIR}/../../../../.." && pwd)
LOG_ROOT=${MCIT_LOG_ROOT:-$HARD_PATH/logs/InternVL/HiDe}
mkdir -p "$LOG_ROOT/wrappers"

TRAIN_SUFFIX=""
DATA_SUFFIX=""
if [ "${MCIT_USE_SMOKE:-0}" = "1" ]; then
    TRAIN_SUFFIX="_smoke"
    DATA_SUFFIX="-smoke"
fi

maybe_eval() {
    if [ "${MCIT_SKIP_EVAL:-0}" = "1" ]; then
        return 0
    fi
    bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh "$1"
}

bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/ImageNet-R${DATA_SUFFIX}.json \
    $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/train/task1${TRAIN_SUFFIX}.json \
    2>&1 | tee "$LOG_ROOT/wrappers/train_UCIT_task1${TRAIN_SUFFIX}.log"
maybe_eval 1

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/ArxivQA${DATA_SUFFIX}.json \
    $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/train/task2${TRAIN_SUFFIX}.json \
    2>&1 | tee "$LOG_ROOT/wrappers/train_UCIT_task2${TRAIN_SUFFIX}.log"
maybe_eval 2

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/VizWiz${DATA_SUFFIX}.json \
    $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/train/task3${TRAIN_SUFFIX}.json \
    2>&1 | tee "$LOG_ROOT/wrappers/train_UCIT_task3${TRAIN_SUFFIX}.log"
maybe_eval 3

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/IconQA${DATA_SUFFIX}.json \
    $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/train/task4${TRAIN_SUFFIX}.json \
    2>&1 | tee "$LOG_ROOT/wrappers/train_UCIT_task4${TRAIN_SUFFIX}.log"
maybe_eval 4

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math${DATA_SUFFIX}.json \
    $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/train/task5${TRAIN_SUFFIX}.json \
    2>&1 | tee "$LOG_ROOT/wrappers/train_UCIT_task5${TRAIN_SUFFIX}.log"
maybe_eval 5

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/UCIT/Flickr30k${DATA_SUFFIX}.json \
    $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/train/task6${TRAIN_SUFFIX}.json \
    2>&1 | tee "$LOG_ROOT/wrappers/train_UCIT_task6${TRAIN_SUFFIX}.log"
maybe_eval 6
