#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"
MCITLIB_ROOT="$(realpath "$SCRIPT_DIR/../../../../..")"

cd "$PROJECT_ROOT"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

DEFAULT_VISIBLE="${CUDA_VISIBLE_DEVICES:-3,4,5,6}"
CUDA_VISIBLE_DEVICES="$(python3 - "$DEFAULT_VISIBLE" <<'PY'
import sys

visible = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
if len(visible) < 4:
    raise SystemExit(f"full DCL needs 4 GPUs, but only got {visible}")
print(",".join(visible[:4]))
PY
)"
export CUDA_VISIBLE_DEVICES

RUN_ID="${RUN_ID:-hide_dcl_full_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$MCITLIB_ROOT/runs/MLLM-DCL/HiDe/$RUN_ID}"
CFG_ROOT="$RUN_ROOT/configs"
CKPT_ROOT="$RUN_ROOT/checkpoints"
RESULT_ROOT="$RUN_ROOT/results/each_dataset"
LOG_DIR="${LOG_DIR:-$MCITLIB_ROOT/logs/MLLM-DCL/full}"

mkdir -p "$CFG_ROOT/train" "$CFG_ROOT/eval" "$CKPT_ROOT" "$RESULT_ROOT" "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$RUN_ID.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Run root: $RUN_ROOT"

python3 - "$MCITLIB_ROOT" "$CFG_ROOT" "$CKPT_ROOT" "$RESULT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
cfg_root = pathlib.Path(sys.argv[2])
ckpt_root = pathlib.Path(sys.argv[3])
result_root = pathlib.Path(sys.argv[4])

train_src = root / "configs" / "train_configs" / "HiDe" / "LLaVA" / "MLLM-DCL" / "train"
eval_src = root / "configs" / "train_configs" / "HiDe" / "LLaVA" / "MLLM-DCL" / "eval"
tasks = [
    ("task1", "RS"),
    ("task2", "Med"),
    ("task3", "AD"),
    ("task4", "Sci"),
    ("task5", "Fin"),
]

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")

for idx, (task_name, _) in enumerate(tasks, start=1):
    train_cfg = json.load((train_src / f"{task_name}.json").open())
    train_cfg["output_dir"] = str(ckpt_root / f"Task{idx}_llava_lora")
    if idx > 1:
        train_cfg["previous_model"] = str(ckpt_root / f"Task{idx-1}_llava_lora")
    dump(cfg_root / "train" / f"{task_name}.json", train_cfg)

    eval_cfg = json.load((eval_src / f"{task_name}.json").open())
    eval_cfg["model_path"] = str(ckpt_root / f"Task{idx}_llava_lora")
    eval_cfg["result_path"] = str(result_root)
    dump(cfg_root / "eval" / f"{task_name}.json", eval_cfg)
PY

MODEL_CFG="$MCITLIB_ROOT/configs/model_configs/llava.json"
DATA_ROOT="$MCITLIB_ROOT/configs/data_configs/MLLM-DCL"

bash scripts/MCITlib/Train/Task1.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/RS.json" \
    "$CFG_ROOT/train/task1.json"
bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/RS.json" \
    "$CFG_ROOT/eval/task1.json"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/Med.json" \
    "$CFG_ROOT/train/task2.json"
bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/Med.json" \
    "$CFG_ROOT/eval/task2.json"
bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/RS.json" \
    "$CFG_ROOT/eval/task2.json"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/AD.json" \
    "$CFG_ROOT/train/task3.json"
bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/Med.json" \
    "$CFG_ROOT/eval/task3.json"
bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/RS.json" \
    "$CFG_ROOT/eval/task3.json"
bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/AD.json" \
    "$CFG_ROOT/eval/task3.json"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/Sci.json" \
    "$CFG_ROOT/train/task4.json"
bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/AD.json" \
    "$CFG_ROOT/eval/task4.json"
bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/RS.json" \
    "$CFG_ROOT/eval/task4.json"
bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/Med.json" \
    "$CFG_ROOT/eval/task4.json"
bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/Sci.json" \
    "$CFG_ROOT/eval/task4.json"

bash scripts/MCITlib/Train/Taskn.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/Fin.json" \
    "$CFG_ROOT/train/task5.json"
bash scripts/MCITlib/Eval_MLLM_DCL/eval_ad.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/AD.json" \
    "$CFG_ROOT/eval/task5.json"
bash scripts/MCITlib/Eval_MLLM_DCL/eval_rs.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/RS.json" \
    "$CFG_ROOT/eval/task5.json"
bash scripts/MCITlib/Eval_MLLM_DCL/eval_med.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/Med.json" \
    "$CFG_ROOT/eval/task5.json"
bash scripts/MCITlib/Eval_MLLM_DCL/eval_sci.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/Sci.json" \
    "$CFG_ROOT/eval/task5.json"
bash scripts/MCITlib/Eval_MLLM_DCL/eval_fin.sh \
    "$MODEL_CFG" \
    "$DATA_ROOT/Fin.json" \
    "$CFG_ROOT/eval/task5.json"
