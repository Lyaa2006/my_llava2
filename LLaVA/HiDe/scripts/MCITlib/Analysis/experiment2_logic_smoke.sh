#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"

cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-$MCITLIB_ROOT/docs/experiment2_l_focus_logic_smoke}"
TASK_IDS="${TASK_IDS:-1 2 3}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-32}"
DESCRIPTION_HIDDEN_LAYER="${DESCRIPTION_HIDDEN_LAYER:--2}"
DESCRIPTION_MAX_TOKENS="${DESCRIPTION_MAX_TOKENS:-56}"
LATE_TASK_START="${LATE_TASK_START:-3}"
LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs/experiment2}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/experiment2_l_focus_logic_smoke_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "TASK_IDS=$TASK_IDS"
echo "SAMPLES_PER_TASK=$SAMPLES_PER_TASK"
echo "DESCRIPTION_HIDDEN_LAYER=$DESCRIPTION_HIDDEN_LAYER"
echo "DESCRIPTION_MAX_TOKENS=$DESCRIPTION_MAX_TOKENS"
echo "LATE_TASK_START=$LATE_TASK_START"
echo "LOG_FILE=$LOG_FILE"

"$PYTHON_BIN" scripts/MCITlib/Analysis/analyze_l_focus_logic.py \
    --output-dir "$OUTPUT_DIR" \
    --task-ids $TASK_IDS \
    --samples-per-task "$SAMPLES_PER_TASK" \
    --description-hidden-layer "$DESCRIPTION_HIDDEN_LAYER" \
    --description-max-tokens "$DESCRIPTION_MAX_TOKENS" \
    --late-task-start "$LATE_TASK_START" \
    2>&1 | tee "$LOG_FILE"
