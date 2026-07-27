#!/usr/bin/env bash
# 从仓库根目录运行:  bash scripts/train_meca.sh
set -ex

export CUDA_VISIBLE_DEVICES=5,6  # 设备在此统一控制

LR=1e-4
NUM_GPUS=2
LORA_RANK=8
LORA_ALPHA=32
LORA_DROPOUT=0.1
MAX_SOURCE_LEN=1100
MAX_TARGET_LEN=128
DEV_BATCH_SIZE=1
GRAD_ACCUMULATION_STEPS=4
NUM_TRAIN_EPOCHS=3

BASE_MODEL_PATH="Huggingface/chatglm3-6b-base"
TRAIN_FILE="./mpf_llm/dataset/finetune_prompt_data/MECES_multimodal_train_finetune.json"
VIDEO_FEATURES="./mpf_llm/dataset/multimodal_features/video_features.pt"
AUDIO_FEATURES="./mpf_llm/dataset/multimodal_features/audio_features.pt"
OUTPUT_DIR="./results/model_checkpoint"
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

mkdir -p "$OUTPUT_DIR"

torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT -m mpf_llm.train \
      --train_format input-output \
      --train_file "$TRAIN_FILE" \
      --video_features_path "$VIDEO_FEATURES" \
      --audio_features_path "$AUDIO_FEATURES" \
      --model_name_or_path "$BASE_MODEL_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --lora_rank $LORA_RANK \
      --lora_alpha $LORA_ALPHA \
      --lora_dropout $LORA_DROPOUT \
      --max_source_length $MAX_SOURCE_LEN \
      --max_target_length $MAX_TARGET_LEN \
      --per_device_train_batch_size $DEV_BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUMULATION_STEPS \
      --num_train_epochs $NUM_TRAIN_EPOCHS \
      --logging_steps 1 \
      --save_strategy "epoch" \
      --learning_rate $LR \
      --remove_unused_columns False \
      --ddp_find_unused_parameters False \
      --fp16 \
      --gradient_checkpointing 2>&1 | tee "${OUTPUT_DIR}/train.log"
