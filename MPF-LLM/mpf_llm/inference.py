"""
运行
scripts/infer_meca.sh
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import random
import re

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from mpf_llm.data import MULTIMODAL_PLACEHOLDER, load_multimodal_features
from mpf_llm.models import MPF_LLM
from mpf_llm.trainer import FUSION_WEIGHTS_NAME


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y"}


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="THUDM/chatglm3-6b-base", help="base LLM 路径/标识")
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--LoRA", type=str2bool, default=True)
    p.add_argument("--lora_path", type=str, required=True, help="LoRA 适配器目录(含 checkpoint)")
    p.add_argument("--fusion_weights_path", type=str, default=None,help="融合模块权重目录，默认与 lora_path 相同")
    p.add_argument("--video_features_path", type=str, required=True)
    p.add_argument("--audio_features_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--test_finetune_data_path", type=str, required=True, help="推理输入(含 context 的 json)")
    p.add_argument("--ref_data_path", type=str, required=True, help="参考标签数据，用于写回预测")
    p.add_argument("--save_pred_data_path", type=str, required=True, help="预测结果保存路径")
    p.add_argument("--raw_pred_data_path", type=str, default=None, help="可选：保存每条原始文本预测")
    args = p.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.fusion_weights_path is None:
        args.fusion_weights_path = args.lora_path
    return args


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_feature_pairs(image_features, audio_features, key_id_list):
    """按 key 顺序返回 [video, audio, video, audio, ...]。"""
    tensors = []
    for key in key_id_list:
        tensors.append(image_features[key].squeeze(0))
        tensors.append(audio_features[key].squeeze(0))
    return tensors


def build_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [MULTIMODAL_PLACEHOLDER]}
    )

    base_model = AutoModel.from_pretrained(args.model, trust_remote_code=True)

    embed_rows = base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embed_rows:
        print(f"Resizing token embeddings: {embed_rows} -> {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))
    else:
        print(f"跳过 resize：embedding 行数 {embed_rows} >= 词表大小 {len(tokenizer)}")

    base_model = base_model.to(args.device)

    if args.LoRA:
        print(f"Loading LoRA adapters from {args.lora_path} ...")
        base_model = PeftModel.from_pretrained(base_model, args.lora_path).merge_and_unload()
        print("LoRA adapters merged.")

    model = MPF_LLM(base_model)

    fusion_path = os.path.join(args.fusion_weights_path, FUSION_WEIGHTS_NAME)
    if not os.path.exists(fusion_path):
        raise FileNotFoundError(f"Fusion weights not found at {fusion_path}")
    print(f"Loading fusion weights from {fusion_path}")
    model.fusion_model.load_state_dict(torch.load(fusion_path, map_location=args.device))

    model = model.to(args.device).eval()
    return tokenizer, model


@torch.no_grad()
def generate_responses(args, tokenizer, model, image_features, audio_features):
    multimodal_token_id = tokenizer.convert_tokens_to_ids(MULTIMODAL_PLACEHOLDER)

    with open(args.test_finetune_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    responses = []
    for data in tqdm(test_data, desc="Generating", unit="ex"):
        inputs = tokenizer(data["context"], return_tensors="pt").to(args.device)
        input_ids = inputs["input_ids"]

        indices = [i for i, tid in enumerate(input_ids[0].tolist()) if tid == multimodal_token_id]
        if not indices:
            raise AssertionError("样本中未找到多模态占位符 token！")

        feats = read_feature_pairs(image_features, audio_features, data["multimodal_features_key_list"])

        # 逐对融合成伪 token
        fused_tokens = []
        for i in range(0, len(feats) - 1, 2):
            video = feats[i].unsqueeze(0).to(args.device)
            audio = feats[i + 1].unsqueeze(0).to(args.device)
            fused_tokens.append(model.fusion_model(video, audio).squeeze(0))

        if len(fused_tokens) != len(indices):
            raise AssertionError(f"融合 token 数({len(fused_tokens)})与占位符数({len(indices)})不一致")

        multimodal_embeds = torch.stack(fused_tokens, dim=0)
        multimodal_indices = torch.tensor(indices, dtype=torch.long, device=args.device)

        output = model.backbone.generate(
            input_ids=input_ids,
            multimodal_embeddings=multimodal_embeds,
            multimodal_indices=multimodal_indices,
            max_length=input_ids.shape[-1] + args.max_new_tokens,
            use_cache=True,
        )
        text = tokenizer.decode(output[0, input_ids.shape[-1]:].cpu(), skip_special_tokens=True)
        responses.append(text)

    return responses


def parse_cause(pred: str):
    """从模型输出解析 (原因 utterance 编号列表, 概括文本)；解析失败返回空。"""
    numbers = list(map(int, re.findall(r"U(\d+)", pred)))
    m = re.search(r"\],(.+)", pred)
    summary = m.group(1).strip() if m else ""
    return numbers, summary


def meces_pred(args, tokenizer, model, image_features, audio_features):
    responses = generate_responses(args, tokenizer, model, image_features, audio_features)

    if args.raw_pred_data_path:
        with open(args.raw_pred_data_path, "w", encoding="utf-8") as f:
            f.write("\n".join(responses))

    with open(args.ref_data_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    idx = 0
    for gold in gold_data:
        for turn in gold["dialog"]:
            emotion = turn["Emotion"]
            if emotion == "Neutral" or turn["Cause_utterance"] == ["无法标注"]:
                continue
            numbers, summary = parse_cause(responses[idx])
            turn["Cause_utterance"] = numbers
            turn["Cause_summary"] = [summary]
            idx += 1

    os.makedirs(os.path.dirname(args.save_pred_data_path), exist_ok=True)
    with open(args.save_pred_data_path, "w", encoding="utf-8") as f:
        json.dump(gold_data, f, indent=6, ensure_ascii=False)
    print(f"Saved predictions to {args.save_pred_data_path}")


def main():
    set_seed(42)
    args = build_args()
    image_features, audio_features = load_multimodal_features(
        args.video_features_path, args.audio_features_path
    )
    tokenizer, model = build_model(args)
    meces_pred(args, tokenizer, model, image_features, audio_features)


if __name__ == "__main__":
    main()
