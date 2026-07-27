#!/usr/bin/env bash
# 从仓库根目录运行:  bash scripts/infer_meca.sh
set -ex

export CUDA_VISIBLE_DEVICES=6

BASE_MODEL_PATH="/data/wjk/chatglm3-6b-base"
LORA_PATH="./results/model_checkpoint/checkpoint-1581"   # 含 LoRA 适配器与 fusion_module.pt
VIDEO_FEATURES="./mpf_llm/dataset/multimodal_features/video_features.pt"
AUDIO_FEATURES="./mpf_llm/dataset/multimodal_features/audio_features.pt"
TEST_FILE="./mpf_llm/dataset/finetune_prompt_data/MECES_multimodal_test_finetune.json"
REF_FILE="./mpf_llm/dataset/MECESD_test.json"
SAVE_PATH="./results/pred/MECES_pred.json"

python -m mpf_llm.inference \
      --model "$BASE_MODEL_PATH" \
      --LoRA True \
      --lora_path "$LORA_PATH" \
      --video_features_path "$VIDEO_FEATURES" \
      --audio_features_path "$AUDIO_FEATURES" \
      --ref_data_path "$REF_FILE" \
      --test_finetune_data_path "$TEST_FILE" \
      --save_pred_data_path "$SAVE_PATH" \
      --max_new_tokens 128

