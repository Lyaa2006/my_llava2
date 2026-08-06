#!/bin/bash

################## VICUNA ##################
PROMPT_VERSION=v1
MODEL_VERSION="vicuna-7b-v1.5"
################## VICUNA ##################

MODEL_CONFIG=$1
DATA_CONFIG=$2
TRAIN_CONFIG=$3

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

RANK=$(read_config "$TRAIN_CONFIG" rank)
EXPERT=$(read_config "$TRAIN_CONFIG" expert_num)
MODEL_NAME=$(read_config "$MODEL_CONFIG" model_name)
MM_PROJECTOR=$(read_config "$MODEL_CONFIG" mm_projector)
DATA_PATH=$(read_config "$DATA_CONFIG" train_path)
IMAGE=$(read_config "$DATA_CONFIG" train_folder)
VISION_TOWER=$(read_config "$MODEL_CONFIG" vision_tower)
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

GPU_NUM=${GPU_NUM_OVERRIDE:-$(read_config "$TRAIN_CONFIG" gpu_num)}
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    DEEPSPEED_GPU_LIST=$(python3 - "$CUDA_VISIBLE_DEVICES" "$GPU_NUM" <<'PY'
import sys

visible = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
gpu_num = int(sys.argv[2])
if len(visible) < gpu_num:
    raise SystemExit(
        f"Requested gpu_num={gpu_num}, but CUDA_VISIBLE_DEVICES only exposes {len(visible)} GPU(s): {visible}"
    )
print(",".join(visible[:gpu_num]))
PY
)
else
    DEEPSPEED_GPU_LIST=""
    for i in $(seq 0 $((GPU_NUM-1))); do
        DEEPSPEED_GPU_LIST+="$i,"
    done
    DEEPSPEED_GPU_LIST=${DEEPSPEED_GPU_LIST%,}
fi

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

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="Llama-2-7b-chat-hf"
################## LLaMA-2 ##################

EXTRA_ARGS=""
if [ "$MAX_STEPS" -gt 0 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --max_steps $MAX_STEPS"
fi

env -u CUDA_VISIBLE_DEVICES deepspeed --include "localhost:$DEEPSPEED_GPU_LIST" --master_port "${MASTER_PORT:-9001}" llava/train/train_mem_MOE.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r $RANK --lora_alpha $((RANK * 2)) --mm_projector_lr 2e-5 \
    --expert_num $EXPERT \
    --model_name_or_path $MODEL_NAME \
    --pretrain_mm_mlp_adapter $MM_PROJECTOR \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --image_folder $IMAGE \
    --vision_tower $VISION_TOWER \
    --text_tower $VISION_TOWER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
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
    $EXTRA_ARGS
