#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEFAULT_ENV_PY="/home/lyaa/miniconda3/envs/MCITlib_copy/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_ENV_PY" ]]; then
    PYTHON_BIN="$DEFAULT_ENV_PY"
  else
    PYTHON_BIN="python3"
  fi
fi
BENCHMARKS="${BENCHMARKS:-ucit acl dcl}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-64}"
SEED="${SEED:-7}"
PROTOCOL_DIR="${PROTOCOL_DIR:-$REPO_ROOT/configs/probe_configs/stage1_protocol}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/docs/stage1_protocol_results}"
MODEL_NAME="${MODEL_NAME:-llava}"
LAYER_COUNT="${LAYER_COUNT:-32}"
B1_LOW="${B1_LOW:-}"
B1_HIGH="${B1_HIGH:-}"
B2_LOW="${B2_LOW:-}"
B2_HIGH="${B2_HIGH:-}"

log() {
  printf '[stage1] %s\n' "$*"
}

require_boundaries() {
  if [[ -z "$B1_LOW" || -z "$B1_HIGH" || -z "$B2_LOW" || -z "$B2_HIGH" ]]; then
    return 1
  fi
  return 0
}

mkdir -p "$PROTOCOL_DIR" "$RESULTS_DIR"

log "1/4 build native probe pool"
"$PYTHON_BIN" scripts/stage1_build_probe_pool.py \
  --benchmarks $BENCHMARKS \
  --max-samples-per-task "$MAX_SAMPLES_PER_TASK" \
  --seed "$SEED" \
  --output-dir "$PROTOCOL_DIR"

log "2/4 build canonical probe pool"
"$PYTHON_BIN" scripts/stage1_build_canonical_pool.py \
  --source-pool "$PROTOCOL_DIR/native_probe_pool.jsonl" \
  --output-dir "$PROTOCOL_DIR" \
  --seed "$SEED"

log "3/4 build split manifest"
"$PYTHON_BIN" scripts/stage1_build_splits.py \
  --input-jsonl "$PROTOCOL_DIR/canonical_probe.jsonl" \
  --output-dir "$PROTOCOL_DIR/splits"

if require_boundaries; then
  log "4/4 build intervention plan"
  "$PYTHON_BIN" scripts/stage1_plan_interventions.py \
    --model "$MODEL_NAME" \
    --layer-count "$LAYER_COUNT" \
    --b1-low "$B1_LOW" \
    --b1-high "$B1_HIGH" \
    --b2-low "$B2_LOW" \
    --b2-high "$B2_HIGH" \
    --output-dir "$PROTOCOL_DIR/interventions"
else
  log "4/4 skip intervention plan: set B1_LOW/B1_HIGH/B2_LOW/B2_HIGH to generate it"
fi

log "done"
log "protocol: $PROTOCOL_DIR"
log "results:  $RESULTS_DIR"
