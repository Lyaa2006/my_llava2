#!/bin/bash

set -euo pipefail

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-__EXTERNAL_ROOT__/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

read_config() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json, sys
path, key = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    print(json.load(f)[key])
PY
}

TASK="AD"
GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
STAGE=$(read_config "$TRAIN_CONFIG" stage)
MODELPATH=$(read_config "$TRAIN_CONFIG" model_path)
MODELBASE=$(read_config "$MODEL_CONFIG" model_name)
DATA_PATH=$(read_config "$DATA_CONFIG" test_path)
IMAGE=$(read_config "$DATA_CONFIG" test_folder)
RESULT_PATH=$(read_config "$TRAIN_CONFIG" result_path)

gpu_list=""
for ((i=0; i<GPU_NUM; i++)); do
    gpu_list+="$i,"
done
gpu_list=${gpu_list%,}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$gpu_list}"

IFS=',' read -ra GPULIST <<< "$CUDA_VISIBLE_DEVICES"
CHUNKS=${#GPULIST[@]}

RESULT_DIR="$RESULT_PATH/$TASK"
mkdir -p "$RESULT_DIR/$STAGE"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} "$PYTHON_BIN" -m llava.eval.CoIN.model_ai2d \
        --model-path "$MODELPATH" \
        --model-base "$MODELBASE" \
        --question-file "$DATA_PATH" \
        --image-folder "$IMAGE" \
        --answers-file "$RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

"$PYTHON_BIN" -m llava.eval.CoIN.eval_ai2d \
    --annotation-file "$DATA_PATH" \
    --result-file "$output_file" \
    --output-dir "$RESULT_DIR/$STAGE" \

# __EXTERNAL_ROOT__/miniconda3/envs/coin/bin/python -m llava.eval.LLaVA.CoIN.create_prompt \
#     --rule ./ETrain/Eval/LLaVA/CoIN/rule.json \
#     --questions ./playground/Instructions_Original/ScienceQA/test.json \
#     --results $output_file \
