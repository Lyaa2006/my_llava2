#!/bin/bash

HARD_PATH=${HARD_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}

bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train/task1.json

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/ArxivQA.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train/task2.json

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/VizWiz.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train/task3.json

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/IconQA.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train/task4.json

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train/task5.json

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/llava.json \
    $HARD_PATH/configs/data_configs/UCIT/Flickr30k.json \
    $HARD_PATH/configs/train_configs/MR-LoRA/LLaVA/UCIT/train/task6.json
bash scripts/MCITlib/Eval_UCIT/Eval_ucit.sh 6
