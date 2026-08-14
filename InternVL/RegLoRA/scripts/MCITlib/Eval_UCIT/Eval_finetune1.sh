#!/bin/bash

set -euo pipefail

TASK_ID=$1
HARD_PATH=${HARD_PATH:-/mnt/lyaa/MCITlib}
DATA_CONFIG_DIR=${DATA_CONFIG_DIR:-$HARD_PATH/configs/data_configs/UCIT}
EVAL_CONFIG_ROOT=${EVAL_CONFIG_ROOT:-$HARD_PATH/configs/train_configs/RegLoRA/InternVL/UCIT}
DATA_SUFFIX=${DATA_SUFFIX:-}
MODEL_CONFIG=$HARD_PATH/configs/model_configs/internvl.json

eval_with_config() {
    local eval_script=$1
    local data_name=$2
    local task_id=$3
    local data_config=$DATA_CONFIG_DIR/${data_name}${DATA_SUFFIX}.json
    local eval_config=$EVAL_CONFIG_ROOT/eval/task${task_id}.json

    bash "$eval_script" "$MODEL_CONFIG" "$data_config" "$eval_config"
}

if [ "$TASK_ID" == "1" ]; then
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_imagenet.sh ImageNet-R 1
elif [ "$TASK_ID" == "2" ]; then
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_imagenet.sh ImageNet-R 2
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh ArxivQA 2
elif [ "$TASK_ID" == "3" ]; then
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_imagenet.sh ImageNet-R 3
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh ArxivQA 3
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh VizWiz 3
elif [ "$TASK_ID" == "4" ]; then
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_imagenet.sh ImageNet-R 4
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh ArxivQA 4
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh VizWiz 4
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_iconqa.sh IconQA 4
elif [ "$TASK_ID" == "5" ]; then
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh ArxivQA 5
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_imagenet.sh ImageNet-R 5
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh VizWiz 5
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_iconqa.sh IconQA 5
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_clevr.sh CLEVR-Math 5
else
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_imagenet.sh ImageNet-R 6
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh ArxivQA 6
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh VizWiz 6
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_iconqa.sh IconQA 6
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_clevr.sh CLEVR-Math 6
    eval_with_config scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh Flickr30k 6
fi
