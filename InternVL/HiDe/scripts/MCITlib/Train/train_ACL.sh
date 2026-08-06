#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HARD_PATH=$(cd "${SCRIPT_DIR}/../../../../.." && pwd)

bash scripts/MCITlib/Train/Task1.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/OCR.json \
    $HARD_PATH/configs/train_configs/HiDe/InternVL/MLLM-ACL/train/task1.json
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 1

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/Math.json \
    $HARD_PATH/configs/train_configs/HiDe/InternVL/MLLM-ACL/train/task2.json
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 2

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/VP.json \
    $HARD_PATH/configs/train_configs/HiDe/InternVL/MLLM-ACL/train/task3.json
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 3

bash scripts/MCITlib/Train/Taskn.sh \
    $HARD_PATH/configs/model_configs/internvl.json \
    $HARD_PATH/configs/data_configs/MLLM-ACL/APP.json \
    $HARD_PATH/configs/train_configs/HiDe/InternVL/MLLM-ACL/train/task4.json
bash scripts/MCITlib/Eval_MLLM_ACL/Eval_finetune1.sh 4
