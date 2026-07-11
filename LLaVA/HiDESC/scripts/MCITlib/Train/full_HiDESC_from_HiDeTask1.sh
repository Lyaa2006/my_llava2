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
export DESCRIPTION_CACHE_MODEL_SOURCE="${DESCRIPTION_CACHE_MODEL_SOURCE:-base}"

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
    "gpu_num": 4,
    "rank": 96,
    "expert_num": 6,
    "epoch": 1,
    "batch_size": 2,
    "grad_acc": 8,
    "lr": 2e-4,
    "save_steps": 50000,
    "model_max_length": 2048,
    "dataloader_num_workers": 4,
    "description_hidden_layer": -2,
    "description_max_tokens": 32,
    "description_focus_weight": 0.2,
    "description_energy_weight": 1e-4,
    "description_energy_margin": 30.0,
    "standard_ce_weight": 3.0,
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
        "gpu_num": 4,
        "stage": f"HiDESC-task{tid}-full-{run_id}",
        "model_path": os.path.join(run_root, f"Task{tid}_llava_lora"),
        "result_path": result_root,
        "text_tower": "/mnt/lyaa/my_llava/clip-vit-large-patch14-336",
        "num_task": tid,
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
            f"reference_description_cache_{cache_source}_{cache_tag}",
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
