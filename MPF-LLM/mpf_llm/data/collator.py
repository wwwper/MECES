"""
多模态数据整理器 (DataCollator)。
"""

from dataclasses import dataclass
from typing import Any, List

import torch


@dataclass
class MultimodalDataCollator:
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: List[dict]) -> dict:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            ids, lab = f["input_ids"], f["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(lab + [self.label_pad_token_id] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            # 多模态原始特征维度不一致，不在此填充，直接以列表透传给模型
            "raw_multimodal_features": [f["raw_multimodal_features"] for f in features],
            "modality_types": [f["modality_types"] for f in features],
            # 每样本一个未填充索引张量；模型内部再统一填充对齐
            "multimodal_indices": [f["multimodal_indices"] for f in features],
        }
