#!/bin/bash

set -euo pipefail

TASK_ID=$1
SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
HARD_PATH=$(cd "$SCRIPT_ROOT/../.." && pwd)
DATA_SUFFIX=""
EVAL_SUFFIX=""

if [ "${MCIT_USE_SMOKE:-0}" = "1" ]; then
    DATA_SUFFIX="-smoke"
    EVAL_SUFFIX="_smoke"
fi

cd "$SCRIPT_ROOT"

if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task1${EVAL_SUFFIX}.json
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task2${EVAL_SUFFIX}.json
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task2${EVAL_SUFFIX}.json
elif [ "$TASK_ID" == "3" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task3${EVAL_SUFFIX}.json
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task3${EVAL_SUFFIX}.json
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/VizWiz${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task3${EVAL_SUFFIX}.json
elif [ "$TASK_ID" == "4" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task4${EVAL_SUFFIX}.json
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task4${EVAL_SUFFIX}.json
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/VizWiz${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task4${EVAL_SUFFIX}.json
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/IconQA${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task4${EVAL_SUFFIX}.json
elif [ "$TASK_ID" == "5" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task5${EVAL_SUFFIX}.json
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task5${EVAL_SUFFIX}.json
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/VizWiz${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task5${EVAL_SUFFIX}.json
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/IconQA${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task5${EVAL_SUFFIX}.json
    bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task5${EVAL_SUFFIX}.json
else
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task6${EVAL_SUFFIX}.json
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task6${EVAL_SUFFIX}.json
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/VizWiz${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task6${EVAL_SUFFIX}.json
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/IconQA${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task6${EVAL_SUFFIX}.json
    bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task6${EVAL_SUFFIX}.json
    bash scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/Flickr30k${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/ModalPrompt/InternVL/UCIT/eval/task6${EVAL_SUFFIX}.json
fi
