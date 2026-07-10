#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../..")"

cd "$PROJECT_ROOT"

GPU_IDS="${GPU_IDS:-0,1}"
SOURCE_ROOT="${SOURCE_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
if [ -x "/home/lyaa/miniconda3/envs/MCITlib/bin/python3.1" ]; then
    PYTHON_BIN="${PYTHON_BIN:-/home/lyaa/miniconda3/envs/MCITlib/bin/python3.1}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [ -z "$SOURCE_ROOT" ] || [ -z "$OUTPUT_ROOT" ]; then
    echo "SOURCE_ROOT and OUTPUT_ROOT must be set." >&2
    exit 1
fi

LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/convert_hidecl_to_hidesc_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"
echo "SOURCE_ROOT=$SOURCE_ROOT"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "GPU_IDS=$GPU_IDS"
echo "PYTHON_BIN=$PYTHON_BIN"

"$PYTHON_BIN" scripts/MCITlib/convert_hidecl_to_hidesc_checkpoints.py \
    --source-root "$SOURCE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --gpu-ids "$GPU_IDS" \
    "$@"
