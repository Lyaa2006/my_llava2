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

MODEL_CONFIG="${MODEL_CONFIG:-configs/model_configs/llava.json}"
PROTOCOL_DIR="${PROTOCOL_DIR:-$REPO_ROOT/configs/probe_configs/stage1_b1}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/docs/stage1_b1_results/$(basename "$MODEL_CONFIG" .json)}"
DEVICE="${DEVICE:-cuda}"
MODEL_FAMILY="${MODEL_FAMILY:-auto}"
if [[ "$MODEL_FAMILY" == "auto" ]]; then
  if [[ "$MODEL_CONFIG" == *internvl* ]]; then
    MODEL_FAMILY="internvl"
  else
    MODEL_FAMILY="llava"
  fi
fi
CONV_MODE="${CONV_MODE:-llava_v1}"
if [[ "$MODEL_FAMILY" == "internvl" && "$CONV_MODE" == "llava_v1" ]]; then
  CONV_MODE="internvl_zh"
fi

SEED="${SEED:-7}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-64}"
BENCHMARKS="${BENCHMARKS:-ucit acl dcl}"
MAX_SPLITS="${MAX_SPLITS:-0}"
SPLIT_TYPE_FILTER="${SPLIT_TYPE_FILTER:-leave_one_dataset_out}"

# b1-only boundary controls
BOUNDARY_STRATEGY="${BOUNDARY_STRATEGY:-b1_only_crossover}"
B1_SEARCH_LOW="${B1_SEARCH_LOW:-10}"
B1_SEARCH_HIGH="${B1_SEARCH_HIGH:-20}"
TRANSITION_MARGIN="${TRANSITION_MARGIN:-0.03}"
TRANSITION_MIN_RUN="${TRANSITION_MIN_RUN:-3}"
MIN_TRANSITION_SUPPORT="${MIN_TRANSITION_SUPPORT:-0.01}"
LATE_PREFERENCE="${LATE_PREFERENCE:-0.01}"
MIN_PRE_REASONING_MARGIN="${MIN_PRE_REASONING_MARGIN:--0.20}"
B1_BAND_SCORE_TOLERANCE="${B1_BAND_SCORE_TOLERANCE:-0.03}"
B1_BAND_MIN_WIDTH="${B1_BAND_MIN_WIDTH:-3}"
B1_BAND_MAX_WIDTH="${B1_BAND_MAX_WIDTH:-5}"
STYLE_AGGREGATION="${STYLE_AGGREGATION:-raw}"
CACHE_DIR="${CACHE_DIR:-}"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton}"
export STAGE1_MODEL_FAMILY="$MODEL_FAMILY"
mkdir -p "$TRITON_CACHE_DIR"

SPLIT_TYPE_ARGS=()
if [[ -n "$SPLIT_TYPE_FILTER" ]]; then
  SPLIT_TYPE_ARGS+=(--split-type-filter "$SPLIT_TYPE_FILTER")
fi

CACHE_ARGS=()
if [[ -n "$CACHE_DIR" ]]; then
  CACHE_ARGS+=(--cache-dir "$CACHE_DIR")
fi

log() {
  printf '[stage1-b1] %s\n' "$*"
}

log "build protocol"
PROTOCOL_DIR="$PROTOCOL_DIR" RESULTS_DIR="$RESULTS_DIR" SEED="$SEED" MAX_SAMPLES_PER_TASK="$MAX_SAMPLES_PER_TASK" BENCHMARKS="$BENCHMARKS" bash scripts/run_stage1_protocol.sh

log "analyze canonical probe for b1"
"$PYTHON_BIN" scripts/stage1_analyze_boundaries.py \
  --model-config "$MODEL_CONFIG" \
  --probe-jsonl "$PROTOCOL_DIR/canonical_probe.jsonl" \
  --split-manifest "$PROTOCOL_DIR/splits/split_manifest.json" \
  "${SPLIT_TYPE_ARGS[@]}" \
  --output-dir "$RESULTS_DIR/canonical" \
  --device "$DEVICE" \
  --conv-mode "$CONV_MODE" \
  --style-aggregation "$STYLE_AGGREGATION" \
  --boundary-strategy "$BOUNDARY_STRATEGY" \
  --b1-search-low "$B1_SEARCH_LOW" \
  --b1-search-high "$B1_SEARCH_HIGH" \
  --transition-margin "$TRANSITION_MARGIN" \
  --transition-min-run "$TRANSITION_MIN_RUN" \
  --min-transition-support "$MIN_TRANSITION_SUPPORT" \
  --late-preference "$LATE_PREFERENCE" \
  --min-pre-reasoning-margin "$MIN_PRE_REASONING_MARGIN" \
  --b1-band-score-tolerance "$B1_BAND_SCORE_TOLERANCE" \
  --b1-band-min-width "$B1_BAND_MIN_WIDTH" \
  --b1-band-max-width "$B1_BAND_MAX_WIDTH" \
  "${CACHE_ARGS[@]}" \
  --max-splits "$MAX_SPLITS"

log "summarize"
"$PYTHON_BIN" scripts/stage1_summarize_results.py \
  --input-root "$RESULTS_DIR/canonical" \
  --output-dir "$RESULTS_DIR"

log "done"
log "summary: $RESULTS_DIR/stage1_summary.json"
