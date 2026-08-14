#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$WORKSPACE_ROOT/../..")"
HIDESC_ROOT="$(realpath "$MCITLIB_ROOT/LLaVA/HiDESC")"

if [ "$#" -ne 2 ]; then
    echo "Usage: bash $0 <model_config> <injected_checkpoint_dir>" >&2
    exit 1
fi

MODEL_CONFIG="$1"
MODEL_PATH="$2"

PYTHON_BIN="${PYTHON_BIN:-/home/lyaa/miniconda3/envs/MCITlib/bin/python3}"
TEXT_TOWER="${TEXT_TOWER:-/mnt/lyaa/my_llava/clip-vit-large-patch14-336}"
GPU_NUM="${GPU_NUM:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
RESULT_ROOT="${RESULT_ROOT:-$WORKSPACE_ROOT/results/UCIT/reglora_task6_hidesc_eval}"
ROUTING_CONFIG_PATH="${ROUTING_CONFIG_PATH:-configs/routing_configs/HiDESC/ucit_role_3way_fft_soft.json}"
STAGE1_BAND_SCHEDULE_PATH="${STAGE1_BAND_SCHEDULE_PATH:-configs/routing_configs/HiDESC/internvl_stage1_band_eval_schedule.json}"
TMP_EVAL_CONFIG="${TMP_EVAL_CONFIG:-$WORKSPACE_ROOT/results/UCIT/reglora_task6_hidesc_eval/task6_eval_config.json}"

mkdir -p "$(dirname "$TMP_EVAL_CONFIG")" "$RESULT_ROOT"

if [ ! -f "$MODEL_CONFIG" ]; then
    echo "Missing model config: $MODEL_CONFIG" >&2
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "Missing injected checkpoint dir: $MODEL_PATH" >&2
    exit 1
fi
if [ ! -d "$HIDESC_ROOT" ]; then
    echo "Missing HiDESC project root: $HIDESC_ROOT" >&2
    exit 1
fi

"$PYTHON_BIN" - "$TMP_EVAL_CONFIG" "$GPU_NUM" "$MODEL_PATH" "$RESULT_ROOT" "$TEXT_TOWER" "$ROUTING_CONFIG_PATH" "$STAGE1_BAND_SCHEDULE_PATH" <<'PY'
import json
import sys

cfg_path = sys.argv[1]
gpu_num = int(sys.argv[2])
model_path = sys.argv[3]
result_root = sys.argv[4]
text_tower = sys.argv[5]
routing_config_path = sys.argv[6]
stage1_band_schedule_path = sys.argv[7]

cfg = {
    "gpu_num": gpu_num,
    "stage": "RegLoRA-task6-HiDESC-eval",
    "model_path": model_path,
    "result_path": result_root,
    "text_tower": text_tower,
    "num_task": 6,
    "routing_config_path": routing_config_path,
    "stage1_band_schedule_path": stage1_band_schedule_path,
}
with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(cfg_path)
PY

cat <<EOF
Task6 HiDESC eval interface is ready.

HiDESC runtime:
  HIDESC_ROOT=$HIDESC_ROOT
  PYTHON_BIN=$PYTHON_BIN
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

Eval config:
  TMP_EVAL_CONFIG=$TMP_EVAL_CONFIG

When GPUs are available, run:
  cd "$HIDESC_ROOT"
  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
  bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh "$MODEL_CONFIG" "$MCITLIB_ROOT/configs/data_configs/UCIT/ImageNet-R.json" "$TMP_EVAL_CONFIG"
  bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh "$MODEL_CONFIG" "$MCITLIB_ROOT/configs/data_configs/UCIT/ArxivQA.json" "$TMP_EVAL_CONFIG"
  bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh "$MODEL_CONFIG" "$MCITLIB_ROOT/configs/data_configs/UCIT/VizWiz.json" "$TMP_EVAL_CONFIG"
  bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh "$MODEL_CONFIG" "$MCITLIB_ROOT/configs/data_configs/UCIT/IconQA.json" "$TMP_EVAL_CONFIG"
  bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh "$MODEL_CONFIG" "$MCITLIB_ROOT/configs/data_configs/UCIT/CLEVR-Math.json" "$TMP_EVAL_CONFIG"
  bash scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh "$MODEL_CONFIG" "$MCITLIB_ROOT/configs/data_configs/UCIT/Flickr30k.json" "$TMP_EVAL_CONFIG"

This wrapper only stages the task6-after-training eval entrypoint. It does not start GPU jobs by itself.
EOF
