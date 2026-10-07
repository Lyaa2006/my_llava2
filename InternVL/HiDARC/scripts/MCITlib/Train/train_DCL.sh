#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
WORKSPACE_ROOT="$(realpath "$PROJECT_ROOT/../..")"
CONFIG_ROOT="${CONFIG_ROOT:-$WORKSPACE_ROOT/configs}"
MODEL_CONFIG="${MODEL_CONFIG:-$CONFIG_ROOT/model_configs/internvl.json}"
DATA_ROOT="${DATA_ROOT:-$CONFIG_ROOT/data_configs/MLLM-DCL}"
RUN_ROOT="${RUN_ROOT:-$WORKSPACE_ROOT/checkpoints/MLLM-DCL/InternVL/HiDESC/mainline_$(date +%Y%m%d_%H%M%S)}"
# All five DCL datasets have a validated, base-model cache at B2=29.  Reuse it
# by default so a normal training run never re-extracts descriptions.
DEFAULT_CACHE_ROOT="/home/lyaa/MCITlib_runs/checkpoints/MLLM-DCL/InternVL/HiDESC/description_caches"
CACHE_ROOT="${CACHE_ROOT:-$DEFAULT_CACHE_ROOT}"
GPU_NUM="${GPU_NUM:-2}"
RANK="${RANK:-96}"
EXPERT_NUM="${EXPERT_NUM:-6}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACC="${GRAD_ACC:-8}"
LR="${LR:-1e-4}"
SPECTRAL_PCA_PATH="${SPECTRAL_PCA_PATH:?Set SPECTRAL_PCA_PATH to the fixed InternVL PCA file}"
SPECTRAL_ROUTE_CHANNEL_DIM="${SPECTRAL_ROUTE_CHANNEL_DIM:-512}"
if [ ! -f "$SPECTRAL_PCA_PATH" ]; then
    echo "Missing InternVL spectral PCA file: $SPECTRAL_PCA_PATH" >&2
    exit 1
fi
GLOBAL_BATCH=$((GPU_NUM * BATCH_SIZE * GRAD_ACC))
if [ "$GLOBAL_BATCH" -ne 32 ]; then
    echo "DCL global batch must remain 32 (got $GLOBAL_BATCH)" >&2
    exit 1
fi
export SPECTRAL_PCA_PATH SPECTRAL_ROUTE_CHANNEL_DIM

mkdir -p "$RUN_ROOT/configs"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$PROJECT_ROOT/scripts/zero2.json}"

python3 - "$CACHE_ROOT" "$DATA_ROOT" <<'PY'
import json
import os
import sys

cache_root, data_root = sys.argv[1:]
prompt = (
    "Describe the image using visual evidence: objects, attributes, shapes, "
    "colors, textures, scene context, visible text, and spatial relations."
)
for tag in ("RS", "Med", "AD", "Sci", "Fin"):
    cache_dir = os.path.join(cache_root, f"{tag}_base_b229_expanded_text_v1")
    meta_path = os.path.join(cache_dir, "meta.json")
    data_path = os.path.join(data_root, f"{tag}.json")
    if not os.path.isfile(meta_path):
        raise SystemExit(f"Missing description-cache metadata: {meta_path}")
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    with open(data_path, encoding="utf-8") as handle:
        data_config = json.load(handle)
    train_path = data_config["train_path"]
    with open(train_path, encoding="utf-8") as handle:
        expected = sum("image" in sample for sample in json.load(handle))
    actual = sum(name.endswith(".pt") for name in os.listdir(cache_dir))
    valid = (
        actual == expected
        and meta.get("data_path") == train_path
        and meta.get("description_prompt") == prompt
        and meta.get("description_cache_model_source") == "base"
        and meta.get("description_cache_format") == "expanded_text_v1"
        and meta.get("b2_high_layer") == 29
        and meta.get("description_max_tokens") == 32
    )
    if not valid:
        raise SystemExit(f"Invalid description cache for {tag}: {cache_dir} ({actual}/{expected})")
    print(f"Description cache ready: {tag} {actual}/{expected} {cache_dir}")
PY

python3 - "$CONFIG_ROOT" "$DATA_ROOT" "$RUN_ROOT" "$CACHE_ROOT" "$GPU_NUM" "$RANK" "$EXPERT_NUM" "$BATCH_SIZE" "$GRAD_ACC" "$LR" <<'PY'
import json
import os
import sys

config_root, data_root, run_root, cache_root, gpu_num, rank, expert_num, batch_size, grad_acc, lr = sys.argv[1:]
tasks = [(1, "RS", 1), (2, "Med", 3), (3, "AD", 1), (4, "Sci", 2), (5, "Fin", 1)]
for task_id, tag, epoch in tasks:
    source_path = os.path.join(config_root, "train_configs", "MR-LoRA", "InternVL", "MLLM-DCL", "train", f"task{task_id}.json")
    with open(source_path, encoding="utf-8") as handle:
        config = json.load(handle)
    config.update({
        "gpu_num": int(gpu_num), "rank": int(rank), "expert_num": int(expert_num),
        "cur_task": task_id - 1, "epoch": epoch, "batch_size": int(batch_size),
        "grad_acc": int(grad_acc), "lr": float(lr),
        "output_dir": os.path.join(run_root, f"Task{task_id}_internvl_hidesc"),
        "description_cache_dir": os.path.join(cache_root, f"{tag}_base_b229_expanded_text_v1"),
        "description_cache_model_source": "base", "description_max_tokens": 32,
        "spectral_pca_path": os.environ["SPECTRAL_PCA_PATH"],
        "spectral_route_channel_dim": int(os.environ["SPECTRAL_ROUTE_CHANNEL_DIM"]),
        "description_focus_weight": 0.2, "description_energy_weight": 1e-4,
        "description_energy_margin": 30.0, "b1_low_layer": 12,
        "b1_high_layer": 14, "b1_center_layer": 14, "b2_low_layer": 27,
        "b2_high_layer": 29, "b2_center_layer": 28, "align_band_eta": 0.5,
        "struct_band_eta": 0.35, "struct_band_energy_rho": 1.0,
        "align_loss_weight": 0.01, "standard_ce_weight": 1.0,
        "description_prompt":
        "Describe the image using visual evidence: objects, attributes, shapes, colors, textures, scene context, visible text, and spatial relations.",
    })
    if task_id > 1:
        config["previous_model"] = os.path.join(run_root, f"Task{task_id-1}_internvl_hidesc")
    else:
        config.pop("previous_model", None)
    with open(os.path.join(run_root, "configs", f"task{task_id}.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
PY

cd "$PROJECT_ROOT"
for spec in "1 RS" "2 Med" "3 AD" "4 Sci" "5 Fin"; do
    read -r task_id tag <<< "$spec"
    if [ "$task_id" -eq 1 ]; then
        bash "$SCRIPT_DIR/Task1.sh" "$MODEL_CONFIG" "$DATA_ROOT/$tag.json" "$RUN_ROOT/configs/task${task_id}.json"
    else
        bash "$SCRIPT_DIR/Taskn.sh" "$MODEL_CONFIG" "$DATA_ROOT/$tag.json" "$RUN_ROOT/configs/task${task_id}.json"
    fi
done

echo "InternVL HiDESC MLLM-DCL mainline training completed."
