#!/usr/bin/env bash
set -euo pipefail

# Inference-only detail analysis.  This script intentionally does not train or
# modify any checkpoint; it delegates generation/scoring to the existing UCIT
# offline-cache evaluator with one routing policy selected by DETAIL_POLICY.

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
CONFIG_ROOT="$MCITLIB_ROOT/configs"

DETAIL_POLICY="${DETAIL_POLICY:-intra_band}"
COMPLETED_ROOT="${COMPLETED_ROOT:-/home/lyaa/MCITlib_runs/checkpoints/UCIT/LLaVA/ucit_hidesc_new_strategy_20260822_001334_offline_cache}"
RESULT_ROOT="${RESULT_ROOT:-/home/lyaa/MCITlib_runs/results/UCIT/hidesc_layer_detail_${DETAIL_POLICY}_$(date +%Y%m%d_%H%M%S)}"
MODEL_CONFIG="${MODEL_CONFIG:-$CONFIG_ROOT/model_configs/llava.json}"
TEXT_TOWER="${TEXT_TOWER:-/mnt/lyaa/my_llava/clip-vit-large-patch14-336}"
STAGE1_SCHEDULE="${STAGE1_SCHEDULE:-$CONFIG_ROOT/routing_configs/HiDESC/llava_stage1_band_eval_schedule.json}"
# Detail analysis is intentionally Last-only: Task6 checkpoint evaluated on
# all six UCIT tasks.  Set TASK_IDS explicitly only if a checkpoint sweep is
# desired later.
TASK_IDS="${TASK_IDS:-6}"
LOG_ROOT="${LOG_ROOT:-$MCITLIB_ROOT/logs}"
RUN_ID="${RUN_ID:-hidesc_layer_detail_${DETAIL_POLICY}_$(date +%Y%m%d_%H%M%S)}"

case "$DETAIL_POLICY" in
  intra_band)
    ROUTING_CONFIG_PATH="${ROUTING_CONFIG_PATH:-$CONFIG_ROOT/routing_configs/HiDESC/ucit_role_new_strategy_detail_intra_band.json}"
    ;;
  full_layer)
    ROUTING_CONFIG_PATH="${ROUTING_CONFIG_PATH:-$CONFIG_ROOT/routing_configs/HiDESC/ucit_role_new_strategy_detail_full_layer.json}"
    ;;
  fixed)
    ROUTING_CONFIG_PATH="${ROUTING_CONFIG_PATH:-$CONFIG_ROOT/routing_configs/HiDESC/ucit_role_new_partition_eval_late_role_prototype_only.json}"
    ;;
  *)
    echo "DETAIL_POLICY must be fixed, intra_band, or full_layer; got $DETAIL_POLICY" >&2
    exit 2
    ;;
esac

for required in "$PROJECT_ROOT/LLaVA/HiDESC/scripts/MCITlib/Eval_UCIT/run_offline_hidesc_ucit_full_eval.sh" \
                "$COMPLETED_ROOT" "$MODEL_CONFIG" "$TEXT_TOWER" \
                "$ROUTING_CONFIG_PATH" "$STAGE1_SCHEDULE"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/${RUN_ID}.log") 2>&1

export COMPLETED_ROOT RESULT_ROOT MODEL_CONFIG TEXT_TOWER
export ROUTING_CONFIG_PATH STAGE1_BAND_SCHEDULE_PATH="$STAGE1_SCHEDULE" TASK_IDS RUN_ID
export HIDESC_LOG_EVAL_ROLE_ACTIVATION=1
export HIDESC_LOG_EVAL_ROLE_ACTIVATION_PATH="$RESULT_ROOT/route_diagnostics.jsonl"

echo "DETAIL_POLICY=$DETAIL_POLICY"
echo "COMPLETED_ROOT=$COMPLETED_ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "ROUTING_CONFIG_PATH=$ROUTING_CONFIG_PATH"
echo "STAGE1_SCHEDULE=$STAGE1_SCHEDULE"
echo "TASK_IDS=$TASK_IDS"
echo "This launcher is inference-only; no checkpoint is written."

exec "$PROJECT_ROOT/LLaVA/HiDESC/scripts/MCITlib/Eval_UCIT/run_offline_hidesc_ucit_full_eval.sh"
