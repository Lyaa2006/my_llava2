#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
HARD_PATH="${HARD_PATH:-$MCITLIB_ROOT}"

cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export UCIT_SMOKE=1
export UCIT_SMOKE_EVAL_LIMIT="${UCIT_SMOKE_EVAL_LIMIT:-32}"
export DESCRIPTION_CACHE_MODEL_SOURCE="${DESCRIPTION_CACHE_MODEL_SOURCE:-base}"

RUN_GROUP="${RUN_GROUP:-LFocus_ablation_smoke_$(date +%Y%m%d_%H%M%S)}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-$MCITLIB_ROOT/checkpoints/UCIT/LLaVA/HiDESC/$RUN_GROUP}"
BASE_RESULT_ROOT="${BASE_RESULT_ROOT:-$MCITLIB_ROOT/LLaVA/HiDeCL/results/UCIT/$RUN_GROUP}"
BASE_LOG_DIR="${BASE_LOG_DIR:-$MCITLIB_ROOT/logs/experiment2/$RUN_GROUP}"
MAX_TASK_ID="${MAX_TASK_ID:-3}"
FOCUS_OFF_WEIGHT="${FOCUS_OFF_WEIGHT:-0.0}"
FOCUS_ON_WEIGHT="${FOCUS_ON_WEIGHT:-0.2}"
ANALYSIS_SAMPLES="${ANALYSIS_SAMPLES:-32}"
ANALYSIS_TASK_IDS="${ANALYSIS_TASK_IDS:-1 2 3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$BASE_RUN_ROOT" "$BASE_RESULT_ROOT" "$BASE_LOG_DIR"

HIDE_TASK1_SRC="${HIDE_TASK1_SRC:-/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe/Task1_llava_lora}"
if [ ! -d "$HIDE_TASK1_SRC" ]; then
    echo "Missing HiDe Task1 checkpoint: $HIDE_TASK1_SRC" >&2
    exit 1
fi

task_data_cfg() {
    local tid=$1
    case "$tid" in
        2) echo "$HARD_PATH/configs/data_configs/UCIT/ArxivQA-smoke.json" ;;
        3) echo "$HARD_PATH/configs/data_configs/UCIT/VizWiz-smoke.json" ;;
        4) echo "$HARD_PATH/configs/data_configs/UCIT/IconQA-smoke.json" ;;
        5) echo "$HARD_PATH/configs/data_configs/UCIT/CLEVR-Math-smoke.json" ;;
        6) echo "$HARD_PATH/configs/data_configs/UCIT/Flickr30k-smoke.json" ;;
        *) echo "Unsupported training task id: $tid" >&2; exit 1 ;;
    esac
}

eval_stage() {
    local tid=$1
    local eval_cfg=$2
    local model_cfg="$HARD_PATH/configs/model_configs/llava.json"

    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh "$model_cfg" "$HARD_PATH/configs/data_configs/UCIT/ImageNet-R-smoke.json" "$eval_cfg"
    if [ "$tid" -ge 2 ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh "$model_cfg" "$HARD_PATH/configs/data_configs/UCIT/ArxivQA-smoke.json" "$eval_cfg"
    fi
    if [ "$tid" -ge 3 ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh "$model_cfg" "$HARD_PATH/configs/data_configs/UCIT/VizWiz-smoke.json" "$eval_cfg"
    fi
    if [ "$tid" -ge 4 ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh "$model_cfg" "$HARD_PATH/configs/data_configs/UCIT/IconQA-smoke.json" "$eval_cfg"
    fi
    if [ "$tid" -ge 5 ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh "$model_cfg" "$HARD_PATH/configs/data_configs/UCIT/CLEVR-Math-smoke.json" "$eval_cfg"
    fi
    if [ "$tid" -ge 6 ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh "$model_cfg" "$HARD_PATH/configs/data_configs/UCIT/Flickr30k-smoke.json" "$eval_cfg"
    fi
}

run_single_condition() {
    local focus_weight=$1
    local tag=$2

    local run_id="${RUN_GROUP}_${tag}"
    local run_root="$BASE_RUN_ROOT/$tag"
    local result_root="$BASE_RESULT_ROOT/$tag"
    local cfg_root="$run_root/generated_configs"
    local log_file="$BASE_LOG_DIR/${tag}.log"
    local analysis_dir="$run_root/analysis_l_focus_logic"
    local task1_dst="$run_root/Task1_llava_lora"

    mkdir -p "$run_root" "$result_root" "$cfg_root"

    if [ ! -d "$task1_dst" ]; then
        echo "[$tag] Copying HiDe Task1 checkpoint to $task1_dst"
        cp -a "$HIDE_TASK1_SRC" "$task1_dst"
    else
        echo "[$tag] Reusing copied Task1 checkpoint: $task1_dst"
    fi

    RUN_ID="$run_id" RUN_ROOT="$run_root" RESULT_ROOT="$result_root" CFG_ROOT="$cfg_root" FOCUS_WEIGHT="$focus_weight" MAX_TASK_ID="$MAX_TASK_ID" \
    DESCRIPTION_CACHE_MODEL_SOURCE="$DESCRIPTION_CACHE_MODEL_SOURCE" python3 - <<'PY'
import json
import os

run_id = os.environ["RUN_ID"]
run_root = os.environ["RUN_ROOT"]
result_root = os.environ["RESULT_ROOT"]
cfg_root = os.environ["CFG_ROOT"]
focus_weight = float(os.environ["FOCUS_WEIGHT"])
max_task_id = int(os.environ["MAX_TASK_ID"])
cache_source = os.environ.get("DESCRIPTION_CACHE_MODEL_SOURCE", "base")

common = {
    "gpu_num": 4,
    "rank": 96,
    "expert_num": 6,
    "epoch": 1,
    "batch_size": 1,
    "grad_acc": 1,
    "lr": 2e-4,
    "max_steps": 1,
    "save_steps": 1,
    "model_max_length": 1024,
    "dataloader_num_workers": 0,
    "description_max_tokens": 32,
    "description_focus_weight": focus_weight,
    "description_energy_weight": 1e-4,
    "description_energy_margin": 30.0,
    "b1_low_layer": 15,
    "b1_high_layer": 18,
    "b2_low_layer": 29,
    "b2_high_layer": 31,
    "align_band_eta": 0.5,
    "struct_band_eta": 0.35,
    "struct_band_energy_rho": 1.0,
    "loss_band_ema_gamma": 0.9,
    "loss_band_position_eps": 0.05,
    "standard_ce_weight": 3.0,
}

task_meta = {
    2: ("ArxivQA-smoke", 1),
    3: ("VizWiz-smoke", 2),
    4: ("IconQA-smoke", 3),
    5: ("CLEVR-Math-smoke", 4),
    6: ("Flickr30k-smoke", 5),
}

for tid in range(1, max_task_id + 1):
    eval_cfg = {
        "gpu_num": 4,
        "stage": f"{run_id}-task{tid}",
        "model_path": os.path.join(run_root, f"Task{tid}_llava_lora"),
        "result_path": result_root,
        "text_tower": "/mnt/lyaa/my_llava/clip-vit-large-patch14-336",
        "num_task": 6,
    }
    with open(os.path.join(cfg_root, f"eval_task{tid}.json"), "w") as f:
        json.dump(eval_cfg, f, indent=2)

for tid in range(2, max_task_id + 1):
    cache_tag, cur_task = task_meta[tid]
    prev_dir = os.path.join(run_root, f"Task{tid-1}_llava_lora")
    out_dir = os.path.join(run_root, f"Task{tid}_llava_lora")
    train_cfg = dict(common)
    train_cfg.update({
        "previous_model": prev_dir,
        "output_dir": out_dir,
        "cur_task": cur_task,
        "description_cache_model_source": cache_source,
        "description_cache_dir": os.path.join(
            prev_dir,
            f"reference_description_cache_{cache_source}_{cache_tag}_expanded_text_v1",
        ),
    })
    with open(os.path.join(cfg_root, f"train_task{tid}.json"), "w") as f:
        json.dump(train_cfg, f, indent=2)
PY

    export LOG_FILE="$log_file"
    export LOG_TEE_ACTIVE=1
    exec > >(tee -a "$log_file") 2>&1

    echo "[$tag] RUN_ID=$run_id"
    echo "[$tag] RUN_ROOT=$run_root"
    echo "[$tag] RESULT_ROOT=$result_root"
    echo "[$tag] CFG_ROOT=$cfg_root"
    echo "[$tag] FOCUS_WEIGHT=$focus_weight"

    echo "[$tag] ===== Eval task1 from inherited HiDe Task1 checkpoint ====="
    eval_stage 1 "$cfg_root/eval_task1.json"

    local tid
    for tid in $(seq 2 "$MAX_TASK_ID"); do
        echo "[$tag] ===== Train task${tid} ====="
        bash scripts/MCITlib/Train/Taskn.sh \
            "$HARD_PATH/configs/model_configs/llava.json" \
            "$(task_data_cfg "$tid")" \
            "$cfg_root/train_task${tid}.json"
        echo "[$tag] ===== Eval task${tid} ====="
        eval_stage "$tid" "$cfg_root/eval_task${tid}.json"
    done

    echo "[$tag] ===== Logic analysis ====="
    "$PYTHON_BIN" "$MCITLIB_ROOT/LLaVA/HiDe/scripts/MCITlib/Analysis/analyze_l_focus_logic.py" \
        --checkpoint-root "$run_root" \
        --output-dir "$analysis_dir" \
        --task-ids $ANALYSIS_TASK_IDS \
        --samples-per-task "$ANALYSIS_SAMPLES" \
        --description-max-tokens 56 \
        --late-task-start 3
}

run_single_condition "$FOCUS_OFF_WEIGHT" "focus_off"
run_single_condition "$FOCUS_ON_WEIGHT" "focus_on"

OFF_REPORT="$BASE_RUN_ROOT/focus_off/analysis_l_focus_logic/l_focus_logic_report.json"
ON_REPORT="$BASE_RUN_ROOT/focus_on/analysis_l_focus_logic/l_focus_logic_report.json"
COMPARE_DIR="$BASE_RUN_ROOT/ablation_comparison"

if [ -f "$OFF_REPORT" ] && [ -f "$ON_REPORT" ]; then
    "$PYTHON_BIN" "$MCITLIB_ROOT/LLaVA/HiDe/scripts/MCITlib/Analysis/compare_l_focus_logic_reports.py" \
        --focus-off-report "$OFF_REPORT" \
        --focus-on-report "$ON_REPORT" \
        --output-dir "$COMPARE_DIR"
fi

echo "===== Done ====="
echo "run group: $RUN_GROUP"
echo "base checkpoints: $BASE_RUN_ROOT"
echo "base results: $BASE_RESULT_ROOT"
echo "logs: $BASE_LOG_DIR"
