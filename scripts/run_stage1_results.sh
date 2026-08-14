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
PROTOCOL_DIR="${PROTOCOL_DIR:-$REPO_ROOT/configs/probe_configs/stage1_protocol}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/docs/stage1_protocol_results/$(basename "$MODEL_CONFIG" .json)}"
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
BOUNDARY_STRATEGY="${BOUNDARY_STRATEGY:-ordered}"
SPLIT_TYPE_FILTER="${SPLIT_TYPE_FILTER:-}"
B1_SEARCH_LOW="${B1_SEARCH_LOW:-10}"
B1_SEARCH_HIGH="${B1_SEARCH_HIGH:-20}"
TRANSITION_ANCHOR_OFFSET="${TRANSITION_ANCHOR_OFFSET:-2}"
TRANSITION_MARGIN="${TRANSITION_MARGIN:-0.03}"
TRANSITION_MIN_RUN="${TRANSITION_MIN_RUN:-3}"
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
  printf '[stage1-results] %s\n' "$*"
}

log "build protocol"
SEED="$SEED" MAX_SAMPLES_PER_TASK="$MAX_SAMPLES_PER_TASK" bash scripts/run_stage1_protocol.sh

log "analyze canonical probe"
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
  --transition-anchor-offset "$TRANSITION_ANCHOR_OFFSET" \
  --transition-margin "$TRANSITION_MARGIN" \
  --transition-min-run "$TRANSITION_MIN_RUN" \
  "${CACHE_ARGS[@]}" \
  --max-splits "$MAX_SPLITS"

log "analyze native probe"
"$PYTHON_BIN" scripts/stage1_analyze_boundaries.py \
  --model-config "$MODEL_CONFIG" \
  --probe-jsonl "$PROTOCOL_DIR/native_probe_pool.jsonl" \
  "${SPLIT_TYPE_ARGS[@]}" \
  --output-dir "$RESULTS_DIR/native" \
  --device "$DEVICE" \
  --conv-mode "$CONV_MODE" \
  --style-aggregation "$STYLE_AGGREGATION" \
  --boundary-strategy "$BOUNDARY_STRATEGY" \
  --b1-search-low "$B1_SEARCH_LOW" \
  --b1-search-high "$B1_SEARCH_HIGH" \
  --transition-anchor-offset "$TRANSITION_ANCHOR_OFFSET" \
  --transition-margin "$TRANSITION_MARGIN" \
  --transition-min-run "$TRANSITION_MIN_RUN" \
  "${CACHE_ARGS[@]}" \
  --max-splits "$MAX_SPLITS"

log "summarize"
"$PYTHON_BIN" scripts/stage1_summarize_results.py \
  --input-root "$RESULTS_DIR" \
  --output-dir "$RESULTS_DIR"

log "done"
log "summary: $RESULTS_DIR/stage1_summary.json"
