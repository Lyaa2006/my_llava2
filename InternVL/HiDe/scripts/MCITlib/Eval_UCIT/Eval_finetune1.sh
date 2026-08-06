# #!/bin/bash

TASK_ID=$1
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARD_PATH=$(cd "${SCRIPT_DIR}/../../../../.." && pwd)
LOG_ROOT=${MCIT_LOG_ROOT:-$HARD_PATH/logs/InternVL/HiDe}
mkdir -p "$LOG_ROOT/eval"

CONFIG_SUFFIX=""
DATA_SUFFIX=""
LOG_SUFFIX=""
if [ "${MCIT_USE_SMOKE:-0}" = "1" ]; then
    CONFIG_SUFFIX="_smoke"
    DATA_SUFFIX="-smoke"
    LOG_SUFFIX="_smoke"
fi

run_eval() {
    local log_file=$1
    shift
    "$@" 2>&1 | tee -a "$log_file"
}

LOG_FILE="$LOG_ROOT/eval/Eval_UCIT_task${TASK_ID}${LOG_SUFFIX}.log"

if [ "$TASK_ID" == "1" ]; then
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task1${CONFIG_SUFFIX}.json
elif [ "$TASK_ID" == "2" ]; then
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task2${CONFIG_SUFFIX}.json
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task2${CONFIG_SUFFIX}.json
elif [ "$TASK_ID" == "3" ]; then
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task3${CONFIG_SUFFIX}.json
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task3${CONFIG_SUFFIX}.json
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/VizWiz${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task3${CONFIG_SUFFIX}.json
elif [ "$TASK_ID" == "4" ]; then
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task4${CONFIG_SUFFIX}.json
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task4${CONFIG_SUFFIX}.json
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/VizWiz${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task4${CONFIG_SUFFIX}.json
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/IconQA${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task4${CONFIG_SUFFIX}.json
elif [ "$TASK_ID" == "5" ]; then
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task5${CONFIG_SUFFIX}.json
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task5${CONFIG_SUFFIX}.json
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/VizWiz${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task5${CONFIG_SUFFIX}.json
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/IconQA${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task5${CONFIG_SUFFIX}.json
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task5${CONFIG_SUFFIX}.json
else
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ImageNet-R${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task6${CONFIG_SUFFIX}.json
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/ArxivQA${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task6${CONFIG_SUFFIX}.json
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/VizWiz${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task6${CONFIG_SUFFIX}.json
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/IconQA${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task6${CONFIG_SUFFIX}.json
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/CLEVR-Math${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task6${CONFIG_SUFFIX}.json
    run_eval "$LOG_FILE" bash scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh $HARD_PATH/configs/model_configs/internvl.json $HARD_PATH/configs/data_configs/UCIT/Flickr30k${DATA_SUFFIX}.json $HARD_PATH/configs/train_configs/HiDe/InternVL/UCIT/eval/task6${CONFIG_SUFFIX}.json
fi
