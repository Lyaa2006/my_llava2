#!/bin/bash

set -x

################## VICUNA ##################
PROMPT_VERSION=v1
################## VICUNA ##################

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HIDE_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../../.." && pwd)

read_config() {
    python3 -c "import json; print(json.load(open('$1'))['$2'])"
}

read_config_default() {
    python3 - "$1" "$2" "$3" <<'PY'
import json
import sys

path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    data = json.load(f)
print(data.get(key, default))
PY
}

NNODES=${NNODES:-1}
GPU_NUM=$(read_config "$TRAIN_CONFIG" gpu_num)
RANK=$(read_config "$TRAIN_CONFIG" rank)
EXPERT=$(read_config "$TRAIN_CONFIG" expert_num)
MODEL_NAME=$(read_config "$MODEL_CONFIG" model_name)
MM_PROJECTOR=$(read_config "$MODEL_CONFIG" mm_projector)
DATA_PATH=$(read_config "$DATA_CONFIG" train_path)
IMAGE=$(read_config "$DATA_CONFIG" train_folder)
VISION_TOWER=$(read_config "$MODEL_CONFIG" vision_tower)
CLIP_TOWER=$(read_config "$MODEL_CONFIG" clip_tower)
OUTPUT_DIR=$(read_config "$TRAIN_CONFIG" output_dir)
CUR_TASK=$(read_config "$TRAIN_CONFIG" cur_task)
EPOCH=$(read_config "$TRAIN_CONFIG" epoch)
BATCH_SIZE=$(read_config "$TRAIN_CONFIG" batch_size)
GRAD_ACC=$(read_config "$TRAIN_CONFIG" grad_acc)
LR=$(read_config "$TRAIN_CONFIG" lr)
SAVE_STEPS=$(read_config_default "$TRAIN_CONFIG" save_steps 50000)
MODEL_MAX_LENGTH=$(read_config_default "$TRAIN_CONFIG" model_max_length 2048)
DATALOADER_NUM_WORKERS=$(read_config_default "$TRAIN_CONFIG" dataloader_num_workers 4)
MAX_STEPS=$(read_config_default "$TRAIN_CONFIG" max_steps -1)

GPU_LIST=""
for i in $(seq 0 $((GPU_NUM-1))); do
    GPU_LIST+="$i,"
done
GPU_LIST=${GPU_LIST%,}

if [ -z "${MASTER_PORT:-}" ]; then
    MASTER_PORT=$(python3 - <<'PY'
import socket

s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)
fi

mkdir -p "$OUTPUT_DIR"
LOG_ROOT=${MCIT_LOG_ROOT:-$REPO_ROOT/logs/InternVL/HiDe}
mkdir -p "$LOG_ROOT/train"
TRAIN_LOG_NAME=$(python3 - "$TRAIN_CONFIG" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
print("_".join(path.parts[-4:]).replace(".json", ".log"))
PY
)
CENTRAL_TRAIN_LOG="$LOG_ROOT/train/$TRAIN_LOG_NAME"

EXTRA_ARGS=""
if [ "$MAX_STEPS" -gt 0 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --max_steps $MAX_STEPS"
fi

echo "Begin running..."
torchrun --nnodes=${NNODES} --nproc_per_node=${GPU_NUM} --master_port "${MASTER_PORT:-9001}" llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r $RANK --lora_alpha $((RANK * 2)) \
    --expert_num $EXPERT \
    --model_name_or_path $MODEL_NAME \
    --pretrain_mm_mlp_adapter $MM_PROJECTOR \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --image_folder $IMAGE \
    --vision_tower $VISION_TOWER \
    --clip_vision_tower $CLIP_TOWER \
    --clip_text_tower $CLIP_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -4 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --cur_task $CUR_TASK \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps $GRAD_ACC \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --lazy_preprocess True \
    --report_to none \
    $EXTRA_ARGS \
    | tee "${OUTPUT_DIR}/train.log" "$CENTRAL_TRAIN_LOG"
