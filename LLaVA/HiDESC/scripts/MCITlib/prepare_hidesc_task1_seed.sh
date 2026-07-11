#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../..")"

cd "$PROJECT_ROOT"

SOURCE_ROOT="${SOURCE_ROOT:-/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe}"
SOURCE_CHECKPOINT_NAME="${SOURCE_CHECKPOINT_NAME:-Task1_llava_lora}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MCITLIB_ROOT/checkpoints/UCIT/LLaVA/HiDESC_seed_from_HiDeTask1_$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0}"

if [ -x "/home/lyaa/miniconda3/envs/MCITlib/bin/python3.1" ]; then
    PYTHON_BIN="${PYTHON_BIN:-/home/lyaa/miniconda3/envs/MCITlib/bin/python3.1}"
elif [ -x "/home/lyaa/miniconda3/envs/MCITlib/bin/python3" ]; then
    PYTHON_BIN="${PYTHON_BIN:-/home/lyaa/miniconda3/envs/MCITlib/bin/python3}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/prepare_hidesc_task1_seed_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "SOURCE_ROOT=$SOURCE_ROOT"
echo "SOURCE_CHECKPOINT_NAME=$SOURCE_CHECKPOINT_NAME"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "GPU_IDS=$GPU_IDS"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "LOG_FILE=$LOG_FILE"

"$PYTHON_BIN" scripts/MCITlib/convert_hidecl_to_hidesc_checkpoints.py \
    --source-root "$SOURCE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --gpu-ids "$GPU_IDS" \
    --checkpoint-names "$SOURCE_CHECKPOINT_NAME"

echo
echo "Converted checkpoint:"
echo "$OUTPUT_ROOT/$SOURCE_CHECKPOINT_NAME"
echo
echo "Manifest:"
echo "$OUTPUT_ROOT/hidesc_conversion_manifest.json"
