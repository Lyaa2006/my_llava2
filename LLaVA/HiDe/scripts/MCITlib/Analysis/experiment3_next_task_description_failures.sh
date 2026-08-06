#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/lyaa/MCITlib"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_JSON="${OUTPUT_JSON:-${ROOT_DIR}/docs/experiment3_next_task_description_failures.json}"
REPORT_MD="${REPORT_MD:-${ROOT_DIR}/docs/experiment3_next_task_description_failures.md}"
SAMPLES_PER_TRANSITION="${SAMPLES_PER_TRANSITION:-2}"
SEED="${SEED:-7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
TEMPERATURE="${TEMPERATURE:-0.0}"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" scripts/inspect_hidesc_next_task_descriptions.py \
  --output "${OUTPUT_JSON}" \
  --report-md "${REPORT_MD}" \
  --samples-per-transition "${SAMPLES_PER_TRANSITION}" \
  --seed "${SEED}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}"
