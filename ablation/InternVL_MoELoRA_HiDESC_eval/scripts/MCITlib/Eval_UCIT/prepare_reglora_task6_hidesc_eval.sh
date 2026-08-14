#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$WORKSPACE_ROOT/../..")"

PYTHON_BIN="${PYTHON_BIN:-/home/lyaa/miniconda3/envs/MCITlib_copy/bin/python}"
HIDESC_PYTHON_BIN="${HIDESC_PYTHON_BIN:-/home/lyaa/miniconda3/envs/MCITlib/bin/python3}"
MODEL_CONFIG="${MODEL_CONFIG:-$MCITLIB_ROOT/configs/model_configs/internvl.json}"
DATA_CONFIG_ROOT="${DATA_CONFIG_ROOT:-$MCITLIB_ROOT/configs/data_configs/UCIT}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$MCITLIB_ROOT/checkpoints/UCIT/InternVL/RegLoRA/Task6_internvl_lora}"
PREP_ROOT="${PREP_ROOT:-$WORKSPACE_ROOT/results/UCIT/reglora_task6_hidesc_prep}"
ANCHOR_CACHE_DIR="${ANCHOR_CACHE_DIR:-$PREP_ROOT/anchor_cache}"
ANCHOR_CACHE_PATH="${ANCHOR_CACHE_PATH:-$ANCHOR_CACHE_DIR/anchors.pt}"
INJECTED_CHECKPOINT_DIR="${INJECTED_CHECKPOINT_DIR:-$PREP_ROOT/Task6_internvl_lora_hidesc_ready}"
LOG_DIR="${LOG_DIR:-$WORKSPACE_ROOT/logs}"
RUN_ID="${RUN_ID:-reglora_task6_hidesc_$(date +%Y%m%d_%H%M%S)}"
ROUTING_IMAGE_WEIGHT="${ROUTING_IMAGE_WEIGHT:-0.5}"
ROUTING_TEXT_WEIGHT="${ROUTING_TEXT_WEIGHT:-0.5}"
ROUTING_HISTORY_WEIGHT="${ROUTING_HISTORY_WEIGHT:-0.0}"
ROUTING_TEMPERATURE="${ROUTING_TEMPERATURE:-0.1}"
ROLE_BIRTH_THRESHOLD="${ROLE_BIRTH_THRESHOLD:-0.55}"
ROLE_MEMBER_TOP_K="${ROLE_MEMBER_TOP_K:-2}"
ROUTING_ROLE_PRIOR_WEIGHT="${ROUTING_ROLE_PRIOR_WEIGHT:-0.0}"
ROUTING_ROLE_MEMBER_WEIGHT="${ROUTING_ROLE_MEMBER_WEIGHT:-0.2}"
ROUTING_ROLE_SIZE_PENALTY="${ROUTING_ROLE_SIZE_PENALTY:-0.3}"

mkdir -p "$PREP_ROOT" "$ANCHOR_CACHE_DIR" "$LOG_DIR"

EXTRACTION_MANIFEST_CMD=(
    "$PYTHON_BIN"
    "$SCRIPT_DIR/extract_reglora_ucit_anchor_cache.py"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --output-dir "$ANCHOR_CACHE_DIR"
    --data-config-root "$DATA_CONFIG_ROOT"
    --model-config "$MODEL_CONFIG"
)

INJECTION_CMD=(
    "$PYTHON_BIN"
    "$SCRIPT_DIR/inject_reglora_hidesc_checkpoint.py"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --anchor-cache "$ANCHOR_CACHE_PATH"
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

cat <<EOF
Prepared task6 RegLoRA -> HiDESC eval interface.

Workspace:
  WORKSPACE_ROOT=$WORKSPACE_ROOT
  PREP_ROOT=$PREP_ROOT
  CHECKPOINT_DIR=$CHECKPOINT_DIR
  ANCHOR_CACHE_DIR=$ANCHOR_CACHE_DIR
  ANCHOR_CACHE_PATH=$ANCHOR_CACHE_PATH
  INJECTED_CHECKPOINT_DIR=$INJECTED_CHECKPOINT_DIR

Python:
  PYTHON_BIN=$PYTHON_BIN
  HIDESC_PYTHON_BIN=$HIDESC_PYTHON_BIN

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

Step 1. Write the extraction manifest:
  ${EXTRACTION_MANIFEST_CMD[*]}

Step 2. After anchors.pt is produced, inject HiDESC tensors into the copied task6 checkpoint:
  ${INJECTION_CMD[*]}

Step 3. Run task6-only continual HiDESC eval:
  bash "$SCRIPT_DIR/run_reglora_task6_hidesc_eval.sh" \\
      "$MODEL_CONFIG" \\
      "$INJECTED_CHECKPOINT_DIR"

Notes:
  - Step 1 is executed below because it does not require a GPU.
  - Step 2 and Step 3 are only staged here. They are not started automatically.
EOF

"${EXTRACTION_MANIFEST_CMD[@]}"
