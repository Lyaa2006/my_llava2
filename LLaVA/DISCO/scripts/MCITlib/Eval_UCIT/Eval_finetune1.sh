#!/bin/bash
set -e

TASK_ID=$1
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
HARD_PATH="${HARD_PATH:-$MCITLIB_ROOT}"
if [ -d "$HARD_PATH/configs" ]; then
    CONFIG_ROOT="$HARD_PATH/configs"
else
    CONFIG_ROOT="$HARD_PATH"
fi

cd "$PROJECT_ROOT"

MODEL_CFG="$CONFIG_ROOT/model_configs/llava.json"

if [ "${UCIT_SMOKE:-0}" = "1" ]; then
    DATA_TASK1="$CONFIG_ROOT/data_configs/UCIT/ImageNet-R-smoke.json"
    DATA_TASK2="$CONFIG_ROOT/data_configs/UCIT/ArxivQA-smoke.json"
    DATA_TASK3="$CONFIG_ROOT/data_configs/UCIT/VizWiz-smoke.json"
    DATA_TASK4="$CONFIG_ROOT/data_configs/UCIT/IconQA-smoke.json"
    DATA_TASK5="$CONFIG_ROOT/data_configs/UCIT/CLEVR-Math-smoke.json"
    DATA_TASK6="$CONFIG_ROOT/data_configs/UCIT/Flickr30k-smoke.json"
    EVAL_TASK1="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/eval/task1_smoke.json"
    EVAL_TASK2="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/eval/task2_smoke.json"
    EVAL_TASK3="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/eval/task3_smoke.json"
    EVAL_TASK4="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/eval/task4_smoke.json"
    EVAL_TASK5="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/eval/task5_smoke.json"
    EVAL_TASK6="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/eval/task6_smoke.json"
else
    DATA_TASK1="$CONFIG_ROOT/data_configs/UCIT/ImageNet-R.json"
    DATA_TASK2="$CONFIG_ROOT/data_configs/UCIT/ArxivQA.json"
    DATA_TASK3="$CONFIG_ROOT/data_configs/UCIT/VizWiz.json"
    DATA_TASK4="$CONFIG_ROOT/data_configs/UCIT/IconQA.json"
    DATA_TASK5="$CONFIG_ROOT/data_configs/UCIT/CLEVR-Math.json"
    DATA_TASK6="$CONFIG_ROOT/data_configs/UCIT/Flickr30k.json"
    EVAL_TASK1="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/eval/task1.json"
    EVAL_TASK2="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/eval/task2.json"
    EVAL_TASK3="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/eval/task3.json"
    EVAL_TASK4="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/eval/task4.json"
    EVAL_TASK5="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/eval/task5.json"
    EVAL_TASK6="$CONFIG_ROOT/train_configs/DISCO/LLaVA/UCIT/eval/task6.json"
fi

if [ "$TASK_ID" == "1" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh "$MODEL_CFG" "$DATA_TASK1" "$EVAL_TASK1"
elif [ "$TASK_ID" == "2" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh "$MODEL_CFG" "$DATA_TASK1" "$EVAL_TASK2"
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh "$MODEL_CFG" "$DATA_TASK2" "$EVAL_TASK2"
elif [ "$TASK_ID" == "3" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh "$MODEL_CFG" "$DATA_TASK1" "$EVAL_TASK3"
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh "$MODEL_CFG" "$DATA_TASK2" "$EVAL_TASK3"
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh "$MODEL_CFG" "$DATA_TASK3" "$EVAL_TASK3"
elif [ "$TASK_ID" == "4" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh "$MODEL_CFG" "$DATA_TASK1" "$EVAL_TASK4"
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh "$MODEL_CFG" "$DATA_TASK2" "$EVAL_TASK4"
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh "$MODEL_CFG" "$DATA_TASK3" "$EVAL_TASK4"
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh "$MODEL_CFG" "$DATA_TASK4" "$EVAL_TASK4"
elif [ "$TASK_ID" == "5" ]; then
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh "$MODEL_CFG" "$DATA_TASK1" "$EVAL_TASK5"
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh "$MODEL_CFG" "$DATA_TASK2" "$EVAL_TASK5"
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh "$MODEL_CFG" "$DATA_TASK3" "$EVAL_TASK5"
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh "$MODEL_CFG" "$DATA_TASK4" "$EVAL_TASK5"
    bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh "$MODEL_CFG" "$DATA_TASK5" "$EVAL_TASK5"
else
    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh "$MODEL_CFG" "$DATA_TASK1" "$EVAL_TASK6"
    bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh "$MODEL_CFG" "$DATA_TASK2" "$EVAL_TASK6"
    bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh "$MODEL_CFG" "$DATA_TASK3" "$EVAL_TASK6"
    bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh "$MODEL_CFG" "$DATA_TASK4" "$EVAL_TASK6"
    bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh "$MODEL_CFG" "$DATA_TASK5" "$EVAL_TASK6"
    bash scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh "$MODEL_CFG" "$DATA_TASK6" "$EVAL_TASK6"
fi
