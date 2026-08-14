#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_CONFIG="${MODEL_CONFIG:-configs/model_configs/llava.json}"
MODEL_STEM="$(basename "$MODEL_CONFIG" .json)"

SEEDS="${SEEDS:-7 13 21 29 35 42}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-128}"
BENCHMARKS="${BENCHMARKS:-ucit acl dcl}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/docs/stage1_b1_band_large_20260811/$MODEL_STEM}"
PROTOCOL_ROOT="${PROTOCOL_ROOT:-$REPO_ROOT/configs/probe_configs/stage1_b1_band_large_20260811/$MODEL_STEM}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
PREWARM_CACHE="${PREWARM_CACHE:-1}"
GPU_MEMORY_SOFT_LIMIT_MB="${GPU_MEMORY_SOFT_LIMIT_MB:-50000}"
GPU_INDEX="${GPU_INDEX:-auto}"
LAUNCHER_LOG_DIR="${LAUNCHER_LOG_DIR:-$RESULTS_ROOT/logs}"

log() {
  printf '[stage1-b1-multiseed] %s\n' "$*"
}

choose_gpu() {
  if [[ "$GPU_INDEX" != "auto" ]]; then
    printf '%s\n' "$GPU_INDEX"
    return 0
  fi

  local picked
  picked="$(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      | awk -F', *' -v limit="$GPU_MEMORY_SOFT_LIMIT_MB" '
        {
          idx=$1 + 0
          mem=$2 + 0
          util=$3 + 0
          if (mem <= limit) {
            if (!have_soft || mem < best_soft_mem || (mem == best_soft_mem && util < best_soft_util)) {
              have_soft = 1
              best_soft_idx = idx
              best_soft_mem = mem
              best_soft_util = util
            }
          }
          if (!have_any || mem < best_any_mem || (mem == best_any_mem && util < best_any_util)) {
            have_any = 1
            best_any_idx = idx
            best_any_mem = mem
            best_any_util = util
          }
        }
        END {
          if (have_soft) {
            print best_soft_idx
          } else if (have_any) {
            print best_any_idx
          }
        }'
  )"

  if [[ -z "$picked" ]]; then
    log "failed to pick a GPU from nvidia-smi output"
    return 1
  fi
  printf '%s\n' "$picked"
}

list_cache_sources() {
  local seed="$1"
  local -a same_seed=()
  local -a other_seeds=()
  local path

  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    same_seed+=("$path")
  done < <(find "$REPO_ROOT/docs" -path "*/${MODEL_STEM}/seed${seed}/canonical/cache" | rg '/stage1_b1_' | sort -u)

  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    if [[ "$path" == *"/seed${seed}/canonical/cache" ]]; then
      continue
    fi
    other_seeds+=("$path")
  done < <(find "$REPO_ROOT/docs" -path "*/${MODEL_STEM}/seed*/canonical/cache" | rg '/stage1_b1_' | sort -u)

  printf '%s\n' "${same_seed[@]}" "${other_seeds[@]}"
}

prewarm_cache() {
  local seed="$1"
  local dest_cache="$2"
  local copied_any=0
  local src

  mkdir -p "$dest_cache"
  while IFS= read -r src; do
    [[ -n "$src" ]] || continue
    [[ "$src" != "$dest_cache" ]] || continue
    if [[ -d "$src" ]]; then
      log "seed${seed}: prewarm cache from $src"
      cp -rn "$src/." "$dest_cache/"
      copied_any=1
    fi
  done < <(list_cache_sources "$seed")

  if [[ "$copied_any" -eq 0 ]]; then
    log "seed${seed}: no existing cache sources found"
  fi
}

run_seed() {
  local seed="$1"
  local gpu_index="$2"
  local results_dir="$RESULTS_ROOT/seed${seed}"
  local protocol_dir="$PROTOCOL_ROOT/seed${seed}"
  local cache_dir="$results_dir/canonical/cache"
  local summary_path="$results_dir/canonical/stage1_boundary_summary.json"

  if [[ "$SKIP_COMPLETED" == "1" && -f "$summary_path" ]]; then
    log "seed${seed}: skip completed ($summary_path)"
    return 0
  fi

  mkdir -p "$results_dir" "$protocol_dir"
  if [[ "$PREWARM_CACHE" == "1" ]]; then
    prewarm_cache "$seed" "$cache_dir"
  fi

  log "seed${seed}: launch on GPU ${gpu_index}"
  CUDA_VISIBLE_DEVICES="$gpu_index" \
  SEED="$seed" \
  MODEL_CONFIG="$MODEL_CONFIG" \
  PROTOCOL_DIR="$protocol_dir" \
  RESULTS_DIR="$results_dir" \
  CACHE_DIR="$cache_dir" \
  MAX_SAMPLES_PER_TASK="$MAX_SAMPLES_PER_TASK" \
  BENCHMARKS="$BENCHMARKS" \
  DEVICE="cuda" \
  bash scripts/run_stage1_b1_results.sh
}

main() {
  local gpu_index
  local launcher_log

  mkdir -p "$LAUNCHER_LOG_DIR" "$RESULTS_ROOT" "$PROTOCOL_ROOT"
  launcher_log="$LAUNCHER_LOG_DIR/launcher_$(date +%Y%m%d_%H%M%S).log"
  : > "$launcher_log"

  gpu_index="$(choose_gpu)"
  log "selected GPU ${gpu_index}" | tee -a "$launcher_log"
  log "results root: $RESULTS_ROOT" | tee -a "$launcher_log"
  log "protocol root: $PROTOCOL_ROOT" | tee -a "$launcher_log"
  log "seeds: $SEEDS" | tee -a "$launcher_log"

  local seed
  for seed in $SEEDS; do
    run_seed "$seed" "$gpu_index" 2>&1 | tee -a "$LAUNCHER_LOG_DIR/seed${seed}.log" | tee -a "$launcher_log"
  done
}

main "$@"
