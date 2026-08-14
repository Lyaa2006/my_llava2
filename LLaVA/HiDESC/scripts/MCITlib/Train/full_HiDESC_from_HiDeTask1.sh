#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
HARD_PATH="${HARD_PATH:-$MCITLIB_ROOT}"

cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export DESCRIPTION_CACHE_MODEL_SOURCE="${DESCRIPTION_CACHE_MODEL_SOURCE:-previous}"

RUN_ID="${RUN_ID:-HiDESC_full_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$MCITLIB_ROOT/checkpoints/UCIT/LLaVA/HiDESC/$RUN_ID}"
RESULT_ROOT="${RESULT_ROOT:-$MCITLIB_ROOT/LLaVA/HiDeCL/results/UCIT/$RUN_ID}"
CFG_ROOT="${CFG_ROOT:-$RUN_ROOT/generated_configs}"
LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_ID.log}"

export RUN_ID RUN_ROOT RESULT_ROOT CFG_ROOT LOG_FILE HARD_PATH

mkdir -p "$RUN_ROOT" "$RESULT_ROOT" "$CFG_ROOT" "$LOG_DIR"

export LOG_TEE_ACTIVE=1
exec > >(tee -a "$LOG_FILE") 2>&1

echo "RUN_ID=$RUN_ID"
echo "RUN_ROOT=$RUN_ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "CFG_ROOT=$CFG_ROOT"
echo "LOG_FILE=$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "DESCRIPTION_CACHE_MODEL_SOURCE=$DESCRIPTION_CACHE_MODEL_SOURCE"

HIDE_TASK1_SRC="${HIDE_TASK1_SRC:-/mnt/lyaa/my_llava/checkpoint/UCIT/LLaVA-1.5/HiDe/Task1_llava_lora}"
TASK1_DST="$RUN_ROOT/Task1_llava_lora"

if [ ! -d "$HIDE_TASK1_SRC" ]; then
    echo "Missing HiDe Task1 checkpoint: $HIDE_TASK1_SRC" >&2
    exit 1
fi

if [ ! -d "$TASK1_DST" ]; then
    echo "Copying HiDe Task1 checkpoint to $TASK1_DST"
    cp -a "$HIDE_TASK1_SRC" "$TASK1_DST"
else
    echo "Reusing existing copied Task1 checkpoint: $TASK1_DST"
fi

python3 - <<'PY'
import json
import os

run_id = os.environ["RUN_ID"]
run_root = os.environ["RUN_ROOT"]
result_root = os.environ["RESULT_ROOT"]
cfg_root = os.environ["CFG_ROOT"]
cache_source = os.environ.get("DESCRIPTION_CACHE_MODEL_SOURCE", "base")

common = {
    "gpu_num": int(os.environ.get("FULL_GPU_NUM", 2)),
    "rank": int(os.environ.get("FULL_RANK", 96)),
    "expert_num": int(os.environ.get("FULL_EXPERT_NUM", 6)),
    "epoch": int(os.environ.get("FULL_EPOCH", 1)),
    "batch_size": int(os.environ.get("FULL_BATCH_SIZE", 2)),
    "grad_acc": int(os.environ.get("FULL_GRAD_ACC", 16)),
    "lr": float(os.environ.get("FULL_LR", 2e-4)),
    "save_steps": int(os.environ.get("FULL_SAVE_STEPS", 50000)),
    "model_max_length": int(os.environ.get("FULL_MODEL_MAX_LENGTH", 2048)),
    "dataloader_num_workers": int(os.environ.get("FULL_DATALOADER_NUM_WORKERS", 4)),
    "description_max_tokens": int(os.environ.get("FULL_DESCRIPTION_MAX_TOKENS", 32)),
    "description_focus_weight": float(os.environ.get("FULL_DESCRIPTION_FOCUS_WEIGHT", 0.05)),
    "description_focus_alpha": float(os.environ.get("FULL_DESCRIPTION_FOCUS_ALPHA", 0.5)),
    "description_energy_weight": float(os.environ.get("FULL_DESCRIPTION_ENERGY_WEIGHT", 5e-4)),
    "description_energy_margin": float(os.environ.get("FULL_DESCRIPTION_ENERGY_MARGIN", 30.0)),
    "b1_low_layer": int(os.environ.get("FULL_B1_LOW_LAYER", 15)),
    "b1_high_layer": int(os.environ.get("FULL_B1_HIGH_LAYER", 18)),
    "b2_low_layer": int(os.environ.get("FULL_B2_LOW_LAYER", 29)),
    "b2_high_layer": int(os.environ.get("FULL_B2_HIGH_LAYER", 31)),
    "align_band_eta": float(os.environ.get("FULL_ALIGN_BAND_ETA", 0.5)),
    "struct_band_eta": float(os.environ.get("FULL_STRUCT_BAND_ETA", 0.08)),
    "struct_band_energy_rho": float(os.environ.get("FULL_STRUCT_BAND_ENERGY_RHO", 1.0)),
    "loss_band_ema_gamma": float(os.environ.get("FULL_LOSS_BAND_EMA_GAMMA", 0.9)),
    "loss_band_position_eps": float(os.environ.get("FULL_LOSS_BAND_POSITION_EPS", 0.05)),
    "align_loss_weight": float(os.environ.get("FULL_ALIGN_LOSS_WEIGHT", 0.005)),
    "standard_ce_weight": float(os.environ.get("FULL_STANDARD_CE_WEIGHT", 3.0)),
}

task_meta = {
    2: ("ArxivQA", 1),
    3: ("VizWiz", 2),
    4: ("IconQA", 3),
    5: ("CLEVR-Math", 4),
    6: ("Flickr30k", 5),
}

for tid in range(1, 7):
    eval_cfg = {
        "gpu_num": int(os.environ.get("FULL_GPU_NUM", 2)),
        "stage": f"HiDESC-task{tid}-full-{run_id}",
        "model_path": os.path.join(run_root, f"Task{tid}_llava_lora"),
        "result_path": result_root,
        "text_tower": "/mnt/lyaa/my_llava/clip-vit-large-patch14-336",
        "num_task": 6,
        "routing_config_path": "configs/routing_configs/HiDESC/ucit_role_3way_fft_soft.json",
        "stage1_band_schedule_path": "configs/routing_configs/HiDESC/llava_stage1_band_eval_schedule.json",
    }
    with open(os.path.join(cfg_root, f"eval_task{tid}.json"), "w") as f:
        json.dump(eval_cfg, f, indent=2)

for tid, (cache_tag, cur_task) in task_meta.items():
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

eval_stage() {
    local tid=$1
    local eval_cfg="$CFG_ROOT/eval_task${tid}.json"
    local model_cfg="$HARD_PATH/configs/model_configs/llava.json"

    bash scripts/MCITlib/Eval_UCIT/eval_imagenet.sh "$model_cfg" "$HARD_PATH/configs/data_configs/UCIT/ImageNet-R.json" "$eval_cfg"
    if [ "$tid" -ge 2 ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_arxivqa.sh "$model_cfg" "$HARD_PATH/configs/data_configs/UCIT/ArxivQA.json" "$eval_cfg"
    fi
    if [ "$tid" -ge 3 ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_vizwiz.sh "$model_cfg" "$HARD_PATH/configs/data_configs/UCIT/VizWiz.json" "$eval_cfg"
    fi
    if [ "$tid" -ge 4 ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_iconqa.sh "$model_cfg" "$HARD_PATH/configs/data_configs/UCIT/IconQA.json" "$eval_cfg"
    fi
    if [ "$tid" -ge 5 ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_clevr.sh "$model_cfg" "$HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json" "$eval_cfg"
    fi
    if [ "$tid" -ge 6 ]; then
        bash scripts/MCITlib/Eval_UCIT/eval_flickr30k.sh "$model_cfg" "$HARD_PATH/configs/data_configs/UCIT/Flickr30k.json" "$eval_cfg"
    fi
}

echo "===== Eval task1 from inherited HiDe Task1 checkpoint ====="
eval_stage 1

echo "===== Train task2 ====="
bash scripts/MCITlib/Train/Taskn.sh \
    "$HARD_PATH/configs/model_configs/llava.json" \
    "$HARD_PATH/configs/data_configs/UCIT/ArxivQA.json" \
    "$CFG_ROOT/train_task2.json"
echo "===== Eval task2 ====="
eval_stage 2

echo "===== Train task3 ====="
bash scripts/MCITlib/Train/Taskn.sh \
    "$HARD_PATH/configs/model_configs/llava.json" \
    "$HARD_PATH/configs/data_configs/UCIT/VizWiz.json" \
    "$CFG_ROOT/train_task3.json"
echo "===== Eval task3 ====="
eval_stage 3

echo "===== Train task4 ====="
bash scripts/MCITlib/Train/Taskn.sh \
    "$HARD_PATH/configs/model_configs/llava.json" \
    "$HARD_PATH/configs/data_configs/UCIT/IconQA.json" \
    "$CFG_ROOT/train_task4.json"
echo "===== Eval task4 ====="
eval_stage 4

echo "===== Train task5 ====="
bash scripts/MCITlib/Train/Taskn.sh \
    "$HARD_PATH/configs/model_configs/llava.json" \
    "$HARD_PATH/configs/data_configs/UCIT/CLEVR-Math.json" \
    "$CFG_ROOT/train_task5.json"
echo "===== Eval task5 ====="
eval_stage 5

echo "===== Train task6 ====="
bash scripts/MCITlib/Train/Taskn.sh \
    "$HARD_PATH/configs/model_configs/llava.json" \
    "$HARD_PATH/configs/data_configs/UCIT/Flickr30k.json" \
    "$CFG_ROOT/train_task6.json"
echo "===== Eval task6 ====="
eval_stage 6

echo "===== Done ====="
echo "checkpoints: $RUN_ROOT"
echo "results: $RESULT_ROOT"
echo "configs: $CFG_ROOT"
echo "log: $LOG_FILE"
