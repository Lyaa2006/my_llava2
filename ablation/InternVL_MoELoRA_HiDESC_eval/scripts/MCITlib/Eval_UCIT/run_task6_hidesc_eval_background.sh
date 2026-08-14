#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
HIDESC_ROOT="$MCITLIB_ROOT/LLaVA/HiDESC"
PYTHON_ENV_BIN="${PYTHON_ENV_BIN:-/home/lyaa/miniconda3/envs/MCITlib/bin}"

MODEL_CONFIG="${1:?Usage: $0 <model_config> <eval_config> <log_root>}"
EVAL_CONFIG="${2:?Usage: $0 <model_config> <eval_config> <log_root>}"
LOG_ROOT="${3:?Usage: $0 <model_config> <eval_config> <log_root>}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

mkdir -p "$LOG_ROOT"
exec >> "$LOG_ROOT/all_eval.log" 2>&1

export PATH="$PYTHON_ENV_BIN:$PATH"
export PYTHONPATH="$HIDESC_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TASKS=(
    "ImageNet-R|ImageNet-R|eval_imagenet.sh"
    "ArxivQA|ArxivQA|eval_arxivqa.sh"
    "VizWiz|VizWiz|eval_vizwiz.sh"
    "IconQA|IconQA|eval_iconqa.sh"
    "CLEVR-Math|CLEVR-Math|eval_clevr.sh"
    "Flickr30k|Flickr30k|eval_flickr30k.sh"
)

echo "[$(date '+%F %T')] Starting Task6-after-training HiDESC eval"
echo "MODEL_CONFIG=$MODEL_CONFIG"
echo "EVAL_CONFIG=$EVAL_CONFIG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HIDESC_ROOT=$HIDESC_ROOT"
echo "PYTHON_BIN=$(command -v python3)"

for task_spec in "${TASKS[@]}"; do
    IFS='|' read -r task_name data_name eval_script <<< "$task_spec"
    task_log="$LOG_ROOT/${data_name}.log"
    data_config="$MCITLIB_ROOT/configs/data_configs/UCIT/${data_name}.json"

    echo
    echo "[$(date '+%F %T')] BEGIN $task_name"
    echo "task_log=$task_log"
    echo "data_config=$data_config"

    if ! (
        cd "$HIDESC_ROOT"
        export CUDA_VISIBLE_DEVICES
        bash "$HIDESC_ROOT/scripts/MCITlib/Eval_UCIT/$eval_script" \
            "$MODEL_CONFIG" \
            "$data_config" \
            "$EVAL_CONFIG"
    ) > "$task_log" 2>&1; then
        cat "$task_log"
        echo "[$(date '+%F %T')] FAILED $task_name"
        echo "[$(date '+%F %T')] Stopping to preserve a clearly attributable result set."
        exit 1
    fi

    cat "$task_log"
    echo "[$(date '+%F %T')] END $task_name"
done

echo
echo "[$(date '+%F %T')] ALL TASKS COMPLETED"
