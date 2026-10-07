#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
WORKSPACE_ROOT="$(realpath "$PROJECT_ROOT/../..")"
CONFIG_ROOT="${CONFIG_ROOT:-$WORKSPACE_ROOT/configs}"
MODEL_CONFIG="${MODEL_CONFIG:-$CONFIG_ROOT/model_configs/internvl.json}"
DATA_ROOT="${DATA_ROOT:-$CONFIG_ROOT/data_configs/UCIT}"
RUN_ROOT="${RUN_ROOT:-$WORKSPACE_ROOT/checkpoints/UCIT/InternVL/HiDESC/mainline_$(date +%Y%m%d_%H%M%S)}"
# Reuse the validated full UCIT base-model description caches by default.
DEFAULT_CACHE_ROOT="/home/lyaa/MCITlib/checkpoints/UCIT/InternVL/MoELoRA_HiDESC/description_caches"
CACHE_ROOT="${CACHE_ROOT:-$DEFAULT_CACHE_ROOT}"
GPU_NUM="${GPU_NUM:-2}"
RANK="${RANK:-96}"
EXPERT_NUM="${EXPERT_NUM:-6}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACC="${GRAD_ACC:-8}"
LR="${LR:-5e-5}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-1024}"
DESCRIPTION_FOCUS_WEIGHT="${DESCRIPTION_FOCUS_WEIGHT:-0.05}"
DESCRIPTION_ENERGY_WEIGHT="${DESCRIPTION_ENERGY_WEIGHT:-5e-5}"
DESCRIPTION_ENERGY_MARGIN="${DESCRIPTION_ENERGY_MARGIN:-30.0}"
ALIGN_LOSS_WEIGHT="${ALIGN_LOSS_WEIGHT:-0.005}"
STANDARD_CE_WEIGHT="${STANDARD_CE_WEIGHT:-1.5}"
SPECTRAL_PCA_PATH="${SPECTRAL_PCA_PATH:?Set SPECTRAL_PCA_PATH to the fixed InternVL PCA file}"
SPECTRAL_ROUTE_CHANNEL_DIM="${SPECTRAL_ROUTE_CHANNEL_DIM:-512}"
if [ ! -f "$SPECTRAL_PCA_PATH" ]; then
    echo "Missing InternVL spectral PCA file: $SPECTRAL_PCA_PATH" >&2
    exit 1
fi
GLOBAL_BATCH=$((GPU_NUM * BATCH_SIZE * GRAD_ACC))
if [ "$GLOBAL_BATCH" -ne 32 ]; then
    echo "UCIT global batch must remain 32 (got $GLOBAL_BATCH)" >&2
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
tasks = [
    ("ImageNet-R", "task1_imagenet_r"),
    ("ArxivQA", "task2_arxivqa"),
    ("VizWiz", "task3_vizwiz"),
    ("IconQA", "task4_iconqa"),
    ("CLEVR-Math", "task5_clevr_math"),
    ("Flickr30k", "task6_flickr30k"),
]
for task_name, cache_name in tasks:
    cache_dir = os.path.join(cache_root, cache_name)
    meta_path = os.path.join(cache_dir, "meta.json")
    data_config_path = os.path.join(data_root, f"{task_name}.json")
    if not os.path.isfile(meta_path):
        raise SystemExit(f"Missing description-cache metadata: {meta_path}")
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    with open(data_config_path, encoding="utf-8") as handle:
        data_config = json.load(handle)
    train_path = data_config["train_path"]
    with open(train_path, encoding="utf-8") as handle:
        expected = sum("image" in sample for sample in json.load(handle))
    actual = sum(name.endswith(".pt") for name in os.listdir(cache_dir))
    valid = (
        actual == expected
        and meta.get("data_path") == train_path
        and meta.get("description_prompt") == prompt
        and meta.get("description_cache_model_source", "base") == "base"
        and meta.get("description_cache_format") == "expanded_text_v1"
        and meta.get("b2_high_layer") == 29
        and meta.get("description_max_tokens") == 32
    )
    if not valid:
        raise SystemExit(f"Invalid description cache for {task_name}: {cache_dir} ({actual}/{expected})")
    print(f"Description cache ready: {task_name} {actual}/{expected} {cache_dir}")
PY

python3 - "$CONFIG_ROOT" "$DATA_ROOT" "$RUN_ROOT" "$CACHE_ROOT" "$MODEL_CONFIG" "$GPU_NUM" "$RANK" "$EXPERT_NUM" "$BATCH_SIZE" "$GRAD_ACC" "$LR" "$MODEL_MAX_LENGTH" "$DESCRIPTION_FOCUS_WEIGHT" "$DESCRIPTION_ENERGY_WEIGHT" "$DESCRIPTION_ENERGY_MARGIN" "$ALIGN_LOSS_WEIGHT" "$STANDARD_CE_WEIGHT" <<'PY'
import json
import os
import sys

(
    config_root, data_root, run_root, cache_root, model_config, gpu_num, rank,
    expert_num, batch_size, grad_acc, lr, max_length, focus_weight,
    energy_weight, energy_margin, align_weight, standard_ce_weight,
) = sys.argv[1:]
task_names = ["ImageNet-R", "ArxivQA", "VizWiz", "IconQA", "CLEVR-Math", "Flickr30k"]
cache_names = [
    "task1_imagenet_r",
    "task2_arxivqa",
    "task3_vizwiz",
    "task4_iconqa",
    "task5_clevr_math",
    "task6_flickr30k",
]
epochs = [1] * len(task_names)
for task_id, (name, cache_name, epoch) in enumerate(zip(task_names, cache_names, epochs), 1):
    source_path = os.path.join(config_root, "train_configs", "MR-LoRA", "InternVL", "UCIT", "train", f"task{task_id}.json")
    with open(source_path, encoding="utf-8") as handle:
        source = json.load(handle)
    data_path = os.path.join(data_root, f"{name}.json")
    cache_dir = os.path.join(cache_root, cache_name)
    output_dir = os.path.join(run_root, f"Task{task_id}_internvl_hidesc")
    config = dict(source)
    config.update({
        "gpu_num": int(gpu_num), "rank": int(rank), "expert_num": int(expert_num),
        "cur_task": task_id - 1, "epoch": epoch, "batch_size": int(batch_size),
        "grad_acc": int(grad_acc), "lr": float(lr), "output_dir": output_dir,
        "data_path": data_path, "description_cache_dir": cache_dir,
        "description_cache_model_source": "base", "description_max_tokens": 32,
        "spectral_pca_path": os.environ["SPECTRAL_PCA_PATH"],
        "spectral_route_channel_dim": int(os.environ["SPECTRAL_ROUTE_CHANNEL_DIM"]),
        "description_focus_weight": float(focus_weight), "description_energy_weight": float(energy_weight),
        "description_energy_margin": float(energy_margin), "b1_low_layer": 12,
        "b1_high_layer": 14, "b1_center_layer": 14, "b2_low_layer": 27,
        "b2_high_layer": 29, "b2_center_layer": 28, "align_band_eta": 0.5,
        "struct_band_eta": 0.35, "struct_band_energy_rho": 1.0,
        "align_loss_weight": float(align_weight), "standard_ce_weight": float(standard_ce_weight),
        "model_max_length": int(max_length), "dataloader_num_workers": 0,
        "description_prompt":
        "Describe the image using visual evidence: objects, attributes, shapes, colors, textures, scene context, visible text, and spatial relations.",
    })
    if task_id > 1:
        config["previous_model"] = os.path.join(run_root, f"Task{task_id-1}_internvl_hidesc")
    else:
        config.pop("previous_model", None)
    output_path = os.path.join(run_root, "configs", f"task{task_id}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
PY

cd "$PROJECT_ROOT"
for task_id in 1 2 3 4 5 6; do
    data_name=(ImageNet-R ArxivQA VizWiz IconQA CLEVR-Math Flickr30k)
    if [ "$task_id" -eq 1 ]; then
        bash "$SCRIPT_DIR/Task1.sh" "$MODEL_CONFIG" "$DATA_ROOT/${data_name[$((task_id-1))]}.json" "$RUN_ROOT/configs/task${task_id}.json"
    else
        bash "$SCRIPT_DIR/Taskn.sh" "$MODEL_CONFIG" "$DATA_ROOT/${data_name[$((task_id-1))]}.json" "$RUN_ROOT/configs/task${task_id}.json"
    fi
done

echo "InternVL HiDESC UCIT mainline training completed."
