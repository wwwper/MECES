import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional  #
import numpy as np
import logging
logger = logging.getLogger(__name__)

@dataclass
class MultimodalDataCollator:
    tokenizer: Any
    model: Optional[Any] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id: int = -100  #


    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        # logger.warning(f"features shape: {features[0].get('multimodal_embeddings', None)}")

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # 1. 先处理 labels padding
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                else:
                    feature["labels"] = np.concatenate([feature["labels"], remainder] if padding_side == "right" else [remainder,feature["labels"]]).astype(np.int64)

        # 2. 使用 tokenizer.pad 处理输入token的padding
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"], "labels": f["labels"]} for f in features],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # 3. 处理 multimodal_embeddings
        # 由于没有经过投影层，raw_multimodal_features维度并不一致，且比较复杂，我们不在这里进行填充，直接将它们收集成列表
        batch['raw_multimodal_features'] = [f['raw_multimodal_features'] for f in features]
        batch['modality_types'] = [f['modality_types'] for f in features]

        # 4. 处理 multimodal_indices
        max_mm_tokens = max(f["multimodal_indices"].shape[0] for f in features)
        padded_mm_indices = []
        for f in features:
            indices = f["multimodal_indices"]
            pad_len = max_mm_tokens - indices.shape[0]
            if pad_len > 0:
                if padding_side == "right":
                    padding = torch.full((pad_len,), -1, dtype=indices.dtype)
                    indices = torch.cat([indices, padding], dim=0)
                else:
                    padding = torch.full((pad_len,), -1, dtype=indices.dtype)
                    indices = torch.cat([padding,indices], dim=0)
            padded_mm_indices.append(indices)

        if max_mm_tokens > 0:
            batch["multimodal_indices"] = torch.stack(padded_mm_indices)
        else:
            raise ValueError("No multimodal tokens found in any sample! Ensure 'multimodal_embeddings' and 'multimodal_indices' are correctly populated.")

        # 5. 准备 decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids

        return batch
