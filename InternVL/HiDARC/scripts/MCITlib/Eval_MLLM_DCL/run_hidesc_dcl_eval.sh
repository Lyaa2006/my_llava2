#!/usr/bin/env bash
set -euo pipefail

# Evaluate latest LLaVA HiDESC+DCL checkpoints with the prototype-only role
# constrained late-layer policy. Full mode evaluates each seen dataset after
# every checkpoint; smoke mode evaluates one sample per dataset on Task5 only.
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"
CONFIG_ROOT="$MCITLIB_ROOT/configs"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export PYTHONUNBUFFERED=1

MODE="${1:-full}"
case "$MODE" in
    smoke)
        CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/lyaa/MCITlib_runs/checkpoints/MLLM-DCL/LLaVA/HiDESC/DCL_full_20260823_new_partition_eval}"
        RESULT_ROOT="${RESULT_ROOT:-/home/lyaa/MCITlib_runs/results/MLLM-DCL/LLaVA/HiDESC/dcl_hidesc_smoke_$(date +%Y%m%d_%H%M%S)}"
        TASK_IDS="${TASK_IDS:-5}"
        DATA_SUFFIX="-smoke"
        STAGE_SUFFIX="smoke"
        GPU_NUM=1
        # Each smoke file has one example; use one allowed GPU to avoid an
        # empty second shard in the legacy evaluator.
        export CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-4}"
        ;;
    full)
        CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/lyaa/MCITlib_runs/checkpoints/MLLM-DCL/LLaVA/HiDESC/DCL_full_20260823_new_partition_eval}"
        RESULT_ROOT="${RESULT_ROOT:-/home/lyaa/MCITlib_runs/results/MLLM-DCL/LLaVA/HiDESC/dcl_hidesc_full_$(date +%Y%m%d_%H%M%S)}"
        TASK_IDS="${TASK_IDS:-1 2 3 4 5}"
        DATA_SUFFIX=""
        STAGE_SUFFIX="full"
        GPU_NUM=2
        ;;
    *) echo "Usage: $0 [smoke|full]" >&2; exit 2 ;;
esac

MODEL_CONFIG="${MODEL_CONFIG:-$CONFIG_ROOT/model_configs/llava.json}"
TEXT_TOWER="${TEXT_TOWER:-/mnt/lyaa/my_llava/clip-vit-large-patch14-336}"
ROUTING_CONFIG="${ROUTING_CONFIG:-$CONFIG_ROOT/routing_configs/HiDESC/dcl_role_new_partition_eval_late_role_prototype_only.json}"
STAGE1_SCHEDULE="${STAGE1_SCHEDULE:-$CONFIG_ROOT/routing_configs/HiDESC/llava_stage1_band_eval_schedule.json}"
LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs}"
RUN_ID="${RUN_ID:-dcl_hidesc_${MODE}_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_ID}.log}"
CFG_ROOT="${CFG_ROOT:-/tmp/${RUN_ID}_eval_cfgs}"
mkdir -p "$RESULT_ROOT" "$LOG_DIR" "$CFG_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

for required in "$MODEL_CONFIG" "$TEXT_TOWER" "$ROUTING_CONFIG" "$STAGE1_SCHEDULE" "$CHECKPOINT_ROOT"; do
    [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done

echo "MODE=$MODE"
echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "ROUTING_CONFIG=$ROUTING_CONFIG"
echo "TASK_IDS=$TASK_IDS"

write_eval_config() {
    local task_id="$1" cfg_path="$2"
    python3 - "$task_id" "$cfg_path" <<'PY'
import json, os, sys
task_id, cfg_path = int(sys.argv[1]), sys.argv[2]
cfg = {
    "gpu_num": int(os.environ["GPU_NUM"]),
    "stage": f"HiDESC-DCL-task{task_id}-{os.environ['STAGE_SUFFIX']}",
    "model_path": os.path.join(os.environ["CHECKPOINT_ROOT"], f"Task{task_id}_llava_lora"),
    "result_path": os.environ["RESULT_ROOT"],
    "text_tower": os.environ["TEXT_TOWER"],
    "num_task": 5,
    "routing_config_path": os.environ["ROUTING_CONFIG"],
    "stage1_band_schedule_path": os.environ["STAGE1_SCHEDULE"],
}
with open(cfg_path, "w") as handle:
    json.dump(cfg, handle, indent=2)
PY
}

run_dataset() {
    local task_id="$1" cfg="$2" name eval_script data_cfg
    case "$task_id" in
        1) name=RS; eval_script=eval_rs.sh ;;
        2) name=Med; eval_script=eval_med.sh ;;
        3) name=AD; eval_script=eval_ad.sh ;;
        4) name=Sci; eval_script=eval_sci.sh ;;
        5) name=Fin; eval_script=eval_fin.sh ;;
        *) echo "Unsupported DCL task id: $task_id" >&2; exit 1 ;;
    esac
    data_cfg="$CONFIG_ROOT/data_configs/MLLM-DCL/${name}${DATA_SUFFIX}.json"
    bash "$SCRIPT_DIR/$eval_script" "$MODEL_CONFIG" "$data_cfg" "$cfg"
}

export CHECKPOINT_ROOT RESULT_ROOT TEXT_TOWER ROUTING_CONFIG STAGE1_SCHEDULE STAGE_SUFFIX GPU_NUM

for checkpoint_task in $TASK_IDS; do
    checkpoint_dir="$CHECKPOINT_ROOT/Task${checkpoint_task}_llava_lora"
    [[ -f "$checkpoint_dir/adapter_model.bin" && -f "$checkpoint_dir/non_lora_trainables.bin" ]] || {
        echo "Incomplete checkpoint: $checkpoint_dir" >&2; exit 1;
    }
    cfg="$CFG_ROOT/task${checkpoint_task}.json"
    write_eval_config "$checkpoint_task" "$cfg"
    echo "=== Task${checkpoint_task} checkpoint: evaluating task1-task${checkpoint_task} ==="
    if [[ "$MODE" == smoke ]]; then
        for eval_task in 1 2 3 4 5; do run_dataset "$eval_task" "$cfg"; done
    else
        for eval_task in $(seq 1 "$checkpoint_task"); do run_dataset "$eval_task" "$cfg"; done
    fi
done

echo "HiDESC DCL $MODE evaluation finished. Results: $RESULT_ROOT"
