#!/usr/bin/env bash
set -euo pipefail

# Probe the routing decision for every seen DCL task after Task k training.
# This intentionally performs one-token generation only; it is not an answer
# accuracy evaluation and never modifies the checkpoint.

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common_path_resolver.sh"

PROJECT_ROOT="$HIDESC_PROJECT_ROOT"
REPO_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/lyaa/MCITlib_runs/checkpoints/MLLM-DCL/LLaVA/HiDESC/DCL_full_20260823_new_partition_eval}"
RESULT_ROOT="${RESULT_ROOT:-/home/lyaa/MCITlib_runs/results/MLLM-DCL/LLaVA/HiDESC/dcl_hidesc_role_proto_routing_matrix_$(date +%Y%m%d_%H%M%S)}"
MODEL_BASE="${MODEL_BASE:-/mnt/lyaa/MCITlib/models/LLaVA/llava-v1.5-7b-localclip}"
TEXT_TOWER="${TEXT_TOWER:-/mnt/lyaa/my_llava/clip-vit-large-patch14-336}"
ROUTING_CONFIG="${ROUTING_CONFIG:-$REPO_ROOT/configs/routing_configs/HiDESC/dcl_role_new_partition_eval_late_role_prototype_only.json}"
STAGE1_SCHEDULE="${STAGE1_SCHEDULE:-$REPO_ROOT/configs/routing_configs/HiDESC/llava_stage1_band_eval_schedule.json}"
GPU_LIST="${GPU_LIST:-4,5}"
SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
SEED="${SEED:-20260826}"

if [[ ! -d "$CHECKPOINT_ROOT" ]]; then
    echo "Missing checkpoint root: $CHECKPOINT_ROOT" >&2
    exit 1
fi
for required_path in "$MODEL_BASE" "$TEXT_TOWER" "$ROUTING_CONFIG" "$STAGE1_SCHEDULE"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Missing required path: $required_path" >&2
        exit 1
    fi
done

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ ${#GPUS[@]} -eq 0 ]]; then
    echo "GPU_LIST must contain at least one GPU id." >&2
    exit 1
fi

mkdir -p "$RESULT_ROOT"
printf '%s\n' \
    "checkpoint_root=$CHECKPOINT_ROOT" \
    "routing_config=$ROUTING_CONFIG" \
    "sample_size=$SAMPLE_SIZE" \
    "seed=$SEED" \
    "gpus=$GPU_LIST" > "$RESULT_ROOT/run_config.txt"

# checkpoint_stage dataset expected_expert
JOBS=(
    "1 RS 0"
    "2 RS 0"
    "2 Med 1"
    "3 RS 0"
    "3 Med 1"
    "3 AD 2"
    "4 RS 0"
    "4 Med 1"
    "4 AD 2"
    "4 Sci 3"
    "5 RS 0"
    "5 Med 1"
    "5 AD 2"
    "5 Sci 3"
    "5 Fin 4"
)

run_probe() {
    local gpu="$1"
    local stage="$2"
    local dataset="$3"
    local expected_expert="$4"
    local checkpoint="$CHECKPOINT_ROOT/Task${stage}_llava_lora"
    local data_root="$REPO_ROOT/MLLM/domain/$dataset"
    local output_dir="$RESULT_ROOT/Task${stage}"
    local output_path="$output_dir/${dataset}.json"
    local log_path="$output_dir/${dataset}.log"

    if [[ ! -f "$checkpoint/adapter_model.bin" || ! -f "$checkpoint/non_lora_trainables.bin" ]]; then
        echo "Task${stage}/${dataset}: incomplete checkpoint: $checkpoint" >&2
        return 1
    fi
    if [[ ! -f "$data_root/test.json" ]]; then
        echo "Task${stage}/${dataset}: missing test data: $data_root/test.json" >&2
        return 1
    fi

    mkdir -p "$output_dir"
    echo "[$(date '+%F %T')] GPU${gpu} Task${stage}/${dataset} expected expert ${expected_expert}"
    (
        cd "$PROJECT_ROOT"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
            scripts/MCITlib/Eval_UCIT/probe_vizwiz_late_route.py \
            --model-path "$checkpoint" \
            --model-base "$MODEL_BASE" \
            --question-file "$data_root/test.json" \
            --image-folder "$data_root" \
            --text-tower "$TEXT_TOWER" \
            --num-task "$stage" \
            --routing-config-path "$ROUTING_CONFIG" \
            --stage1-band-schedule-path "$STAGE1_SCHEDULE" \
            --output "$output_path" \
            --sample-size "$SAMPLE_SIZE" \
            --seed "$SEED" \
            --expected-experts "$expected_expert" \
            --max-new-tokens 1
    ) > "$log_path" 2>&1
}

worker() {
    local gpu="$1"
    shift
    local job
    for job in "$@"; do
        read -r stage dataset expected_expert <<< "$job"
        run_probe "$gpu" "$stage" "$dataset" "$expected_expert"
    done
}

worker_jobs=()
for _ in "${GPUS[@]}"; do
    worker_jobs+=("")
done
for index in "${!JOBS[@]}"; do
    worker_index=$((index % ${#GPUS[@]}))
    worker_jobs[$worker_index]+="${JOBS[$index]}|"
done

worker_pids=()
for index in "${!GPUS[@]}"; do
    IFS='|' read -r -a assigned_jobs <<< "${worker_jobs[$index]}"
    filtered_jobs=()
    for job in "${assigned_jobs[@]}"; do
        [[ -n "$job" ]] && filtered_jobs+=("$job")
    done
    worker "${GPUS[$index]}" "${filtered_jobs[@]}" &
    worker_pids+=("$!")
done

printf '%s\n' "${worker_pids[@]}" > "$RESULT_ROOT/worker_pids.txt"
echo "Workers started: ${worker_pids[*]}"

status=0
for worker_pid in "${worker_pids[@]}"; do
    if ! wait "$worker_pid"; then
        status=1
    fi
done

"$PYTHON_BIN" - "$RESULT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("Task*/*.json")):
    payload = json.loads(path.read_text())
    stage = int(path.parent.name[len("Task"):])
    dataset = path.stem
    expected = payload["expected_experts"]
    selected = payload["selected_histogram"]
    total = int(payload["sample_size"])
    correct = sum(int(selected.get(str(expert), 0)) for expert in expected)
    rows.append(
        {
            "checkpoint_stage": stage,
            "dataset": dataset,
            "expected_expert": expected[0] if len(expected) == 1 else expected,
            "sample_size": total,
            "correct_routes": correct,
            "route_accuracy": correct / total if total else 0.0,
            "selected_histogram": selected,
            "selected_role_histogram": payload["selected_role_histogram"],
        }
    )

stage_summary = {}
for stage in sorted({row["checkpoint_stage"] for row in rows}):
    stage_rows = [row for row in rows if row["checkpoint_stage"] == stage]
    total = sum(row["sample_size"] for row in stage_rows)
    correct = sum(row["correct_routes"] for row in stage_rows)
    stage_summary[str(stage)] = {
        "seen_task_count": len(stage_rows),
        "correct_routes": correct,
        "sample_size": total,
        "micro_route_accuracy": correct / total if total else 0.0,
        "macro_route_accuracy": sum(row["route_accuracy"] for row in stage_rows) / len(stage_rows),
    }

summary = {"rows": rows, "stage_summary": stage_summary}
(root / "routing_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = [
    "# DCL HiDESC Routing Matrix",
    "",
    "| Checkpoint | Dataset | Expected expert | Correct / Samples | Route accuracy | Selected experts |",
    "| --- | --- | ---: | ---: | ---: | --- |",
]
for row in rows:
    hist = ", ".join(f"{key}:{value}" for key, value in sorted(row["selected_histogram"].items(), key=lambda item: int(item[0])))
    lines.append(
        f"| Task{row['checkpoint_stage']} | {row['dataset']} | {row['expected_expert']} | "
        f"{row['correct_routes']} / {row['sample_size']} | {row['route_accuracy']:.2%} | {hist} |"
    )
lines.extend(["", "| Checkpoint | Seen tasks | Micro route accuracy | Macro route accuracy |", "| --- | ---: | ---: | ---: |"])
for stage, data in stage_summary.items():
    lines.append(
        f"| Task{stage} | {data['seen_task_count']} | {data['micro_route_accuracy']:.2%} | {data['macro_route_accuracy']:.2%} |"
    )
(root / "routing_summary.md").write_text("\n".join(lines) + "\n")
print(json.dumps(stage_summary, indent=2))
PY

exit "$status"
