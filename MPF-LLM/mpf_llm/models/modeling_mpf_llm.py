"""
MPF-LLM

结构:
    MPF_LLM (Multimodal Prompt Fusion + LLM，backbone 不限于 ChatGLM)
        ├── backbone      : 经过改造、可接收 multimodal_embeddings / multimodal_indices 的完整 LLM
        └── fusion_model  : 多模态多层级融合模块 (MultiModal_MLF)，把每对 (video, audio) 特征融合成 1 个 token

注意 batch:
    本封装对 batch_size == 1 完全正确。当 batch_size > 1 且各样本多模态 token 数量不同时， 需要 backbone 能够忽略 index == -1 的填充位（否则填充位会污染最后一个 token）。
"""

from typing import List

import torch
import torch.nn as nn

from .fusion import MultiModal_MLF


class MPF_LLM(nn.Module):
    """在任意 LLM backbone 外层包一层多模态融合与注入逻辑（backbone 不限于 ChatGLM）。"""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        # backbone: 完整的 LLM（如 ChatGLMForConditionalGeneration），已被改造为可接收
        # multimodal_embeddings / multimodal_indices；其内部自身还含一个 transformer 主干。
        self.backbone = backbone
        # fusion_model: 多模态多尺度融合模块，把每对 (video, audio) 融合成 1 个伪 token。
        self.fusion_model = MultiModal_MLF(
            fusion_input_size=768,
            fusion_output_size=4096,   # 需与 backbone 的 word embedding 维度一致
            video_feat_dim=768,
            audio_feat_dim=1024,
        )

    # ---- 供 Trainer 透传的接口，均委托给 backbone ----
    @property
    def config(self):
        return self.backbone.config

    def gradient_checkpointing_enable(self, **kwargs):
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        self.backbone.enable_input_require_grads()

    def _fuse_one_sample(self, feats: List[torch.Tensor], mods: List[str],
                         n_placeholders: int) -> torch.Tensor:
        """
        feats: [video0, audio0, video1, audio1, ...]，共 2K 个张量
        mods : ['video','audio','video','audio', ...]，共 2K 个
        return: [K, dim]，K 应等于该样本占位符数量 n_placeholders
        """
        fused_tokens = []
        for i in range(0, len(feats) - 1, 2):
            # 借助 modality_types 显式校验成对顺序，而非仅靠隐式假设
            if mods[i] != "video" or mods[i + 1] != "audio":
                raise ValueError(f"模态顺序异常，期望 (video, audio)，实际 ({mods[i]}, {mods[i + 1]})")
            video = feats[i].unsqueeze(0)      # [1, seq, feat]
            audio = feats[i + 1].unsqueeze(0)
            fused = self.fusion_model(video, audio).squeeze(0)  # [dim]
            fused_tokens.append(fused)

        if len(fused_tokens) != n_placeholders:
            raise ValueError(
                f"融合 token 数({len(fused_tokens)})与占位符数({n_placeholders})不一致"
            )
        return torch.stack(fused_tokens, dim=0)  # [K, dim]

    def forward(self,
                input_ids: torch.Tensor,
                labels: torch.Tensor,
                raw_multimodal_features: List[List[torch.Tensor]],
                modality_types: List[List[str]],
                multimodal_indices: List[torch.Tensor],   # 每样本一个未填充的 1D LongTensor
                **kwargs):
        # 1. 逐样本融合，得到 [K_i, dim]
        per_sample_embeds = [
            self._fuse_one_sample(feats, mods, idxs.numel())
            for feats, mods, idxs in zip(raw_multimodal_features, modality_types, multimodal_indices)
        ]

        # 2. 目标 dtype / device 对齐 backbone
        ref_param = next(self.backbone.parameters())
        target_dtype, target_device = ref_param.dtype, ref_param.device

        # 3. 统一右填充 embeds 与 indices（右填充不改变已有 token 位置，indices 依旧有效）
        max_k = max(e.shape[0] for e in per_sample_embeds)
        feat_dim = per_sample_embeds[0].shape[1]

        padded_embeds, padded_indices = [], []
        for embeds, idxs in zip(per_sample_embeds, multimodal_indices):
            idxs = idxs.to(target_device)
            pad = max_k - embeds.shape[0]
            if pad > 0:
                embeds = torch.cat([embeds, embeds.new_zeros(pad, feat_dim)], dim=0)
                # 填充位用 -1 标记，backbone 需忽略（batch_size==1 时不会出现填充位）
                idxs = torch.cat([idxs, torch.full((pad,), -1, dtype=idxs.dtype, device=target_device)])
            padded_embeds.append(embeds)
            padded_indices.append(idxs)

        batch_multimodal_embeds = torch.stack(padded_embeds).to(dtype=target_dtype, device=target_device)
        batch_multimodal_indices = torch.stack(padded_indices)

        # 4. 交给 backbone LLM
        final_inputs = {
            "input_ids": input_ids,
            "labels": labels,
            "multimodal_embeddings": batch_multimodal_embeds,
            "multimodal_indices": batch_multimodal_indices,
            **kwargs,  # attention_mask 等
        }
        return self.backbone(**final_inputs)
