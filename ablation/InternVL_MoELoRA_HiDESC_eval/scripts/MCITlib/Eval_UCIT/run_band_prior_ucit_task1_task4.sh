#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
PYTHON_BIN="${PYTHON_BIN:-/home/lyaa/miniconda3/envs/MCITlib_copy/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/lyaa/MCITlib/checkpoints/UCIT/InternVL/MoELoRA/full_g4567_20260802_220559}"
NATIVE_RESULT_ROOT="${NATIVE_RESULT_ROOT:-/mnt/lyaa/MCITlib/InternVL/MoELoRA/results/UCIT/full_g4567_20260802_220559}"
RESULT_ROOT="${RESULT_ROOT:-$WORKSPACE_ROOT/results/UCIT/band_prior_g4567_20260802_220559}"
MODEL_BASE="${MODEL_BASE:-/mnt/lyaa/MCITlib/models/InternVL/Internvl-chat-7b}"
BAND_PRIOR_CONFIG_PATH="${BAND_PRIOR_CONFIG_PATH:-$WORKSPACE_ROOT/configs/ucit_band_prior_eval.json}"
MODEL_CONFIG="${MODEL_CONFIG:-/mnt/lyaa/MCITlib/configs/model_configs/internvl.json}"

cd "$WORKSPACE_ROOT"
mkdir -p "$RESULT_ROOT"

echo "WORKSPACE_ROOT=$WORKSPACE_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "BAND_PRIOR_CONFIG_PATH=$BAND_PRIOR_CONFIG_PATH"

run_eval() {
    local dataset_name="$1"
    local eval_task_id="$2"
    local data_json="$3"
    local metric_mode="$4"
    local checkpoint_stage="$5"

    local model_path="$CHECKPOINT_ROOT/Task${checkpoint_stage}_internvl_lora"
    local stage_name="MoELoRA-band-prior-task${checkpoint_stage}"
    local result_dir="$RESULT_ROOT/$dataset_name/$stage_name"
    mkdir -p "$result_dir"

    echo
    echo "=== $dataset_name | checkpoint Task${checkpoint_stage} | eval task ${eval_task_id} ==="

    "$PYTHON_BIN" - "$data_json" "$dataset_name" "$eval_task_id" "$model_path" "$result_dir" "$MODEL_BASE" "$BAND_PRIOR_CONFIG_PATH" "$metric_mode" "$CUDA_VISIBLE_DEVICES" <<'PY'
import json
import os
import subprocess
import sys

data_json = sys.argv[1]
dataset_name = sys.argv[2]
eval_task_id = int(sys.argv[3])
model_path = sys.argv[4]
result_dir = sys.argv[5]
model_base = sys.argv[6]
band_prior_config_path = sys.argv[7]
metric_mode = sys.argv[8]
cuda_visible_devices = sys.argv[9]

with open(data_json, "r", encoding="utf-8") as f:
    data_cfg = json.load(f)

question_file = data_cfg["test_path"]
image_folder = data_cfg["test_folder"]
annotation_file = data_cfg.get("anno_path", question_file)
gpu_list = [gpu.strip() for gpu in cuda_visible_devices.split(",") if gpu.strip()]
if not gpu_list:
    gpu_list = ["0"]
num_chunks = len(gpu_list)
chunk_paths = []
procs = []
for chunk_idx, gpu_id in enumerate(gpu_list):
    chunk_file = os.path.join(result_dir, f"{num_chunks}_{chunk_idx}.jsonl")
    chunk_paths.append(chunk_file)
    cmd = [
        sys.executable,
        "-m",
        "llava.eval.CoIN.model_others",
        "--model-path",
        model_path,
        "--model-base",
        model_base,
        "--question-file",
        question_file,
        "--image-folder",
        image_folder,
        "--answers-file",
        chunk_file,
        "--conv-mode",
        "vicuna_v1",
        "--temperature",
        "0",
        "--num_beams",
        "1",
        "--num-chunks",
        str(num_chunks),
        "--chunk-idx",
        str(chunk_idx),
        "--band-prior-config-path",
        band_prior_config_path,
        "--eval-task-id",
        str(eval_task_id),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    procs.append(subprocess.Popen(cmd, env=env))

return_codes = [proc.wait() for proc in procs]
if any(code != 0 for code in return_codes):
    raise RuntimeError(f"Generation failed for {dataset_name} task{eval_task_id}: {return_codes}")

answers_file = os.path.join(result_dir, "merge.jsonl")
with open(answers_file, "w", encoding="utf-8") as out_f:
    for chunk_file in chunk_paths:
        if not os.path.isfile(chunk_file):
            raise FileNotFoundError(f"Missing chunk file: {chunk_file}")
        with open(chunk_file, "r", encoding="utf-8") as in_f:
            out_f.write(in_f.read())

if metric_mode == "caption":
    eval_cmd = [
        sys.executable,
        "-m",
        "llava.eval.CoIN.eval_caption",
        "--annotation-file",
        annotation_file,
        "--result-file",
        answers_file,
        "--output-dir",
        result_dir,
    ]
else:
    eval_cmd = [
        sys.executable,
        "-m",
        "llava.eval.CoIN.eval_deepseek_r1",
        "--annotation-file",
        question_file,
        "--result-file",
        answers_file,
        "--output-dir",
        result_dir,
    ]
subprocess.run(eval_cmd, check=True)
PY
}

run_eval "ImageNet-R" 1 "/mnt/lyaa/MCITlib/configs/data_configs/UCIT/ImageNet-R.json" "accuracy" 1
run_eval "ImageNet-R" 1 "/mnt/lyaa/MCITlib/configs/data_configs/UCIT/ImageNet-R.json" "accuracy" 2
run_eval "ArxivQA" 2 "/mnt/lyaa/MCITlib/configs/data_configs/UCIT/ArxivQA.json" "accuracy" 2
run_eval "ImageNet-R" 1 "/mnt/lyaa/MCITlib/configs/data_configs/UCIT/ImageNet-R.json" "accuracy" 3
run_eval "ArxivQA" 2 "/mnt/lyaa/MCITlib/configs/data_configs/UCIT/ArxivQA.json" "accuracy" 3
run_eval "VizWiz" 3 "/mnt/lyaa/MCITlib/configs/data_configs/UCIT/VizWiz.json" "caption" 3
run_eval "ImageNet-R" 1 "/mnt/lyaa/MCITlib/configs/data_configs/UCIT/ImageNet-R.json" "accuracy" 4
run_eval "ArxivQA" 2 "/mnt/lyaa/MCITlib/configs/data_configs/UCIT/ArxivQA.json" "accuracy" 4
run_eval "VizWiz" 3 "/mnt/lyaa/MCITlib/configs/data_configs/UCIT/VizWiz.json" "caption" 4
run_eval "IconQA" 4 "/mnt/lyaa/MCITlib/configs/data_configs/UCIT/IconQA.json" "accuracy" 4

"$PYTHON_BIN" "$WORKSPACE_ROOT/scripts/MCITlib/Eval_UCIT/summarize_vs_native.py" \
    --native-root "$NATIVE_RESULT_ROOT" \
    --ablation-root "$RESULT_ROOT" \
    --output-json "$RESULT_ROOT/summary_vs_native.json" \
    --output-csv "$RESULT_ROOT/summary_vs_native.csv"

echo
echo "Finished band-prior eval. Summary:"
sed -n '1,40p' "$RESULT_ROOT/summary_vs_native.csv"
