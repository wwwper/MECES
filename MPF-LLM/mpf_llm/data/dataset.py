"""
数据集与预处理工具。

"""

from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

MULTIMODAL_PLACEHOLDER = "<multimodal_placeholder>"


def load_multimodal_features(video_path: str, audio_path: str):
    """显式加载视频/音频特征字典 {key: tensor}。由训练/推理脚本调用一次即可。"""
    return torch.load(video_path), torch.load(audio_path)


def sanity_check(tokens: List[int], target: List[int], tokenizer: PreTrainedTokenizer):
    """打印 input_ids 与 labels 的对齐情况，便于快速核对数据构造是否正确。"""
    assert len(tokens) == len(target), f"length mismatch: {len(tokens)} vs {len(target)}"
    print("Sanity Check >>>>>>>>>>>>>")
    for t, m in zip(tokens, target):
        decoded = (
            tokenizer.tokenizer.index_special_tokens[t]
            if t in tokenizer.tokenizer.index_special_tokens
            else tokenizer.decode([t])
        )
        print("%20s: %6d -> %6d" % (repr(decoded), t, m))
    print("<<<<<<<<<<<<< Sanity Check")


class InputOutputDataset(Dataset):
    """input-output 格式数据集，每条样本产出文本 token 与对应的多模态特征。"""

    def __init__(self, data: List[dict], tokenizer: PreTrainedTokenizer,
                 image_features: Dict, audio_features: Dict,
                 max_source_length: int, max_target_length: int):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.image_features = image_features
        self.audio_features = audio_features
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.multimodal_token_id = tokenizer.convert_tokens_to_ids(MULTIMODAL_PLACEHOLDER)

    def __len__(self):
        return len(self.data)

    def _read_features(self, key_id_list: List) -> List[torch.Tensor]:
        """按 key 顺序读取，每个 key 产出 [video, audio] 两个张量。"""
        tensors = []
        for key in key_id_list:
            tensors.append(self.image_features[key].squeeze(0))
            tensors.append(self.audio_features[key].squeeze(0))
        return tensors

    def __getitem__(self, index: int) -> dict:
        item = self.data[index]

        a_ids = self.tokenizer.encode(item["context"], add_special_tokens=True,
                                      truncation=True, max_length=self.max_source_length)
        b_ids = self.tokenizer.encode(item["target"], add_special_tokens=False,
                                      truncation=True, max_length=self.max_target_length)

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        # 上下文部分不计损失(-100)，仅对 target + eos 计损失
        labels = [-100] * context_length + b_ids + [self.tokenizer.eos_token_id]
        assert len(input_ids) == len(labels)

        # 定位多模态占位符
        indices = [i for i, tid in enumerate(input_ids)
                   if tid == self.multimodal_token_id]
        if len(indices) == 0:
            raise AssertionError("样本中未找到多模态占位符 token！")

        feature_keys = item["multimodal_features_key_list"]
        # 每个 key 对应 1 个占位符（数据集为 one-token 版本），截断不应破坏该对应关系
        if len(indices) != len(feature_keys):
            raise AssertionError(
                f"占位符数({len(indices)})与特征 key 数({len(feature_keys)})不一致，"
                f"可能是 max_source_length 过小导致占位符被截断"
            )

        multimodal_feats = self._read_features(feature_keys)                 # 2K 个张量
        modality_types = ["video" if i % 2 == 0 else "audio"                 # 2K 个标签
                          for i in range(len(feature_keys) * 2)]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "raw_multimodal_features": multimodal_feats,
            "modality_types": modality_types,
            "multimodal_indices": torch.tensor(indices, dtype=torch.long),
        }
