#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"

cd "$PROJECT_ROOT"

if [ -x "/home/lyaa/miniconda3/envs/MCITlib/bin/python3.1" ]; then
    PYTHON_BIN="${PYTHON_BIN:-/home/lyaa/miniconda3/envs/MCITlib/bin/python3.1}"
elif [ -x "/home/lyaa/miniconda3/envs/MCITlib/bin/python3" ]; then
    PYTHON_BIN="${PYTHON_BIN:-/home/lyaa/miniconda3/envs/MCITlib/bin/python3}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/lyaa/my_llava/llava-v1.5-7b}"
VISION_TOWER_PATH="${VISION_TOWER_PATH:-/mnt/lyaa/my_llava/clip-vit-large-patch14-336}"
UCIT_ROOT="${UCIT_ROOT:-/mnt/lyaa/my_llava/UCIT}"
# Source of truth: logs/train_UCIT_full_20260703_211225.log
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe}"
OUTPUT_DIR="${OUTPUT_DIR:-$MCITLIB_ROOT/docs/experiment2_template_alignment_v21}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-64}"
SEED="${SEED:-7}"
DEVICE="${DEVICE:-cuda}"

TASK_SPLIT_GPU3="${TASK_SPLIT_GPU3:-1 2}"
TASK_SPLIT_GPU4="${TASK_SPLIT_GPU4:-3}"
TASK_SPLIT_GPU5="${TASK_SPLIT_GPU5:-4 5}"
TASK_SPLIT_GPU6="${TASK_SPLIT_GPU6:-6}"

LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

export OUTPUT_DIR SAMPLES_PER_TASK SEED

PIDS=()
launch_shard() {
    local gpu="$1"
    local task_ids="$2"
    local shard_dir="$OUTPUT_DIR/shard_gpu${gpu}"
    local shard_log="$LOG_DIR/experiment2_template_alignment_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$shard_dir"
    CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON_BIN" scripts/MCITlib/Analysis/analyze_template_alignment.py \
        --base-model-path "$BASE_MODEL_PATH" \
        --vision-tower-path "$VISION_TOWER_PATH" \
        --ucit-root "$UCIT_ROOT" \
        --checkpoint-root "$CHECKPOINT_ROOT" \
        --output-dir "$shard_dir" \
        --task-ids $task_ids \
        --samples-per-task "$SAMPLES_PER_TASK" \
        --device "$DEVICE" \
        --seed "$SEED" \
        > "$shard_log" 2>&1 &
    PIDS+=("$!")
}

launch_shard 3 "$TASK_SPLIT_GPU3"
launch_shard 4 "$TASK_SPLIT_GPU4"
launch_shard 5 "$TASK_SPLIT_GPU5"
launch_shard 6 "$TASK_SPLIT_GPU6"

for pid in "${PIDS[@]}"; do
    wait "$pid"
done

"$PYTHON_BIN" - <<'PY'
import glob
import importlib.util
import json
import os
from pathlib import Path

mcitlib_root = Path("/mnt/lyaa/MCITlib")
output_dir = Path(os.environ.get("OUTPUT_DIR", str(mcitlib_root / "docs" / "experiment2_template_alignment_v21")))
script_path = mcitlib_root / "LLaVA" / "HiDe" / "scripts" / "MCITlib" / "Analysis" / "analyze_template_alignment.py"

spec = importlib.util.spec_from_file_location("analyze_template_alignment", str(script_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

task_reports = []
merged_config = {}
for path in sorted(glob.glob(str(output_dir / "shard_gpu*" / "template_alignment_report.json"))):
    with open(path, "r", encoding="utf-8") as f:
        shard = json.load(f)
    if not merged_config:
        merged_config = shard.get("config", {})
    task_reports.extend(shard.get("tasks", []))

task_reports.sort(key=lambda item: item["task_id"])
merged = {
    "config": merged_config,
    "tasks": task_reports,
}

report_path = output_dir / "template_alignment_report.json"
figure_path = output_dir / "template_alignment_curves.png"
mod.save_report(str(report_path), merged)
mod.plot_results(str(figure_path), task_reports)
print(f"Saved merged report to: {report_path}")
print(f"Saved merged figure to: {figure_path}")
PY
