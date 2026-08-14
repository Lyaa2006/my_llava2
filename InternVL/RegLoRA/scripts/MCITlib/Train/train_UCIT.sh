#!/bin/bash

set -euo pipefail

HARD_PATH=${HARD_PATH:-/mnt/lyaa/MCITlib}
DATA_CONFIG_DIR=${DATA_CONFIG_DIR:-$HARD_PATH/configs/data_configs/UCIT}
TRAIN_CONFIG_ROOT=${TRAIN_CONFIG_ROOT:-$HARD_PATH/configs/train_configs/RegLoRA/InternVL/UCIT}
DATA_SUFFIX=${DATA_SUFFIX:-}
START_TASK=${START_TASK:-1}
END_TASK=${END_TASK:-6}
MODEL_CONFIG=$HARD_PATH/configs/model_configs/internvl.json

run_task() {
    local task_id=$1
    local data_name=$2
    local train_script=$3
    local data_config=$DATA_CONFIG_DIR/${data_name}${DATA_SUFFIX}.json
    local train_config=$TRAIN_CONFIG_ROOT/train/task${task_id}.json

    bash "$train_script" "$MODEL_CONFIG" "$data_config" "$train_config"
    bash scripts/MCITlib/Eval_UCIT/Eval_finetune1.sh "$task_id"
}

for task_id in $(seq "$START_TASK" "$END_TASK"); do
    case "$task_id" in
        1)
            run_task 1 ImageNet-R scripts/MCITlib/Train/Task1.sh
            ;;
        2)
            run_task 2 ArxivQA scripts/MCITlib/Train/Taskn.sh
            ;;
        3)
            run_task 3 VizWiz scripts/MCITlib/Train/Taskn.sh
            ;;
        4)
            run_task 4 IconQA scripts/MCITlib/Train/Taskn.sh
            ;;
        5)
            run_task 5 CLEVR-Math scripts/MCITlib/Train/Taskn.sh
            ;;
        6)
            run_task 6 Flickr30k scripts/MCITlib/Train/Taskn.sh
            ;;
        *)
            echo "Unsupported task id: $task_id" >&2
            exit 1
            ;;
    esac
done
