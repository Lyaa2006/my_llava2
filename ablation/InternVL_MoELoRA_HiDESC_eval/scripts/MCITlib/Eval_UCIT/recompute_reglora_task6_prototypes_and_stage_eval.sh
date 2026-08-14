#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$WORKSPACE_ROOT/../..")"

PYTHON_BIN="${PYTHON_BIN:-/home/lyaa/miniconda3/envs/MCITlib/bin/python3}"
EXPORT_PY="${EXPORT_PY:-$MCITLIB_ROOT/LLaVA/HiDeRA/scripts/MCITlib/export_ucit_anchors.py}"
INJECT_PY="${INJECT_PY:-$SCRIPT_DIR/inject_reglora_hidesc_checkpoint.py}"
STAGE_EVAL_SH="${STAGE_EVAL_SH:-$SCRIPT_DIR/run_reglora_task6_hidesc_eval.sh}"

MODEL_CONFIG="${MODEL_CONFIG:-$MCITLIB_ROOT/ablation/InternVL_MoELoRA_HiDESC_eval/configs/internvl_anchor_export_proxy.json}"
EVAL_MODEL_CONFIG="${EVAL_MODEL_CONFIG:-$MCITLIB_ROOT/configs/model_configs/internvl.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$MCITLIB_ROOT/checkpoints/UCIT/InternVL/RegLoRA/Task6_internvl_lora}"
PREP_ROOT="${PREP_ROOT:-$WORKSPACE_ROOT/results/UCIT/reglora_task6_hidesc_prep}"
ANCHOR_CACHE_DIR="${ANCHOR_CACHE_DIR:-$PREP_ROOT/anchor_cache}"
INJECTED_CHECKPOINT_DIR="${INJECTED_CHECKPOINT_DIR:-$PREP_ROOT/Task6_internvl_lora_hidesc_ready}"
LOG_DIR="${LOG_DIR:-$WORKSPACE_ROOT/logs}"

GPU_EXTRACT="${GPU_EXTRACT:-0}"
GPU_EVAL="${GPU_EVAL:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROMPT_VERSION="${PROMPT_VERSION:-v1}"
IMAGE_ASPECT_RATIO="${IMAGE_ASPECT_RATIO:-pad}"
MAX_SAMPLES_PER_TASK="${MAX_SAMPLES_PER_TASK:-}"
ROUTING_IMAGE_WEIGHT="${ROUTING_IMAGE_WEIGHT:-0.5}"
ROUTING_TEXT_WEIGHT="${ROUTING_TEXT_WEIGHT:-0.5}"
ROUTING_HISTORY_WEIGHT="${ROUTING_HISTORY_WEIGHT:-0.0}"
ROUTING_TEMPERATURE="${ROUTING_TEMPERATURE:-0.1}"
ROLE_BIRTH_THRESHOLD="${ROLE_BIRTH_THRESHOLD:-0.55}"
ROLE_MEMBER_TOP_K="${ROLE_MEMBER_TOP_K:-2}"
ROUTING_ROLE_PRIOR_WEIGHT="${ROUTING_ROLE_PRIOR_WEIGHT:-0.0}"
ROUTING_ROLE_MEMBER_WEIGHT="${ROUTING_ROLE_MEMBER_WEIGHT:-0.2}"
ROUTING_ROLE_SIZE_PENALTY="${ROUTING_ROLE_SIZE_PENALTY:-0.3}"

mkdir -p "$ANCHOR_CACHE_DIR" "$LOG_DIR"

EXPORT_CMD=(
    "$PYTHON_BIN"
    "$EXPORT_PY"
    --model-config "$MODEL_CONFIG"
    --output-dir "$ANCHOR_CACHE_DIR"
    --prompt-version "$PROMPT_VERSION"
    --model-max-length 2048
    --image-aspect-ratio "$IMAGE_ASPECT_RATIO"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --device cuda
    --dtype fp16
)

if [ -n "$MAX_SAMPLES_PER_TASK" ]; then
    EXPORT_CMD+=(--max-samples-per-task "$MAX_SAMPLES_PER_TASK")
fi

INJECT_CMD=(
    "$PYTHON_BIN"
    "$INJECT_PY"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --anchor-cache "$ANCHOR_CACHE_DIR/anchors.pt"
    --output-dir "$INJECTED_CHECKPOINT_DIR"
    --routing-image-weight "$ROUTING_IMAGE_WEIGHT"
    --routing-text-weight "$ROUTING_TEXT_WEIGHT"
    --routing-history-weight "$ROUTING_HISTORY_WEIGHT"
    --routing-temperature "$ROUTING_TEMPERATURE"
    --role-birth-threshold "$ROLE_BIRTH_THRESHOLD"
    --role-member-top-k "$ROLE_MEMBER_TOP_K"
    --routing-role-prior-weight "$ROUTING_ROLE_PRIOR_WEIGHT"
    --routing-role-member-weight "$ROUTING_ROLE_MEMBER_WEIGHT"
    --routing-role-size-penalty "$ROUTING_ROLE_SIZE_PENALTY"
)

STAGE_CMD=(
    bash "$STAGE_EVAL_SH"
    "$EVAL_MODEL_CONFIG"
    "$INJECTED_CHECKPOINT_DIR"
)

cat <<EOF
Step 1. Recompute UCIT prototype cache for RegLoRA->HiDESC eval:
  CUDA_VISIBLE_DEVICES=$GPU_EXTRACT ${EXPORT_CMD[*]}

Step 2. Inject the recomputed cache into the copied task6 RegLoRA checkpoint:
  ${INJECT_CMD[*]}

Step 3. Stage the task6-only HiDESC eval config and print the final eval commands:
  CUDA_VISIBLE_DEVICES=$GPU_EVAL ${STAGE_CMD[*]}

InternVL role config:
  ROUTING_IMAGE_WEIGHT=$ROUTING_IMAGE_WEIGHT
  ROUTING_TEXT_WEIGHT=$ROUTING_TEXT_WEIGHT
  ROUTING_HISTORY_WEIGHT=$ROUTING_HISTORY_WEIGHT
  ROUTING_TEMPERATURE=$ROUTING_TEMPERATURE
  ROLE_BIRTH_THRESHOLD=$ROLE_BIRTH_THRESHOLD
  ROLE_MEMBER_TOP_K=$ROLE_MEMBER_TOP_K
  ROUTING_ROLE_PRIOR_WEIGHT=$ROUTING_ROLE_PRIOR_WEIGHT
  ROUTING_ROLE_MEMBER_WEIGHT=$ROUTING_ROLE_MEMBER_WEIGHT
  ROUTING_ROLE_SIZE_PENALTY=$ROUTING_ROLE_SIZE_PENALTY
EOF

CUDA_VISIBLE_DEVICES="$GPU_EXTRACT" "${EXPORT_CMD[@]}"
"${INJECT_CMD[@]}"
CUDA_VISIBLE_DEVICES="$GPU_EVAL" "${STAGE_CMD[@]}"
