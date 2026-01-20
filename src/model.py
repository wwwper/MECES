

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class MultiModalConfig:
    """多模态模块配置类"""
    video_feat_dim: int = 768
    audio_feat_dim: int = 1024
    fusion_hidden_dim: int = 768
    output_embed_dim: int = 4096
    num_fusion_scales: int = 3
    # 各个尺度的压缩因子，对应原代码中的 output_size // factor
    scale_factors: Tuple[int, ...] = (8, 32, 16)


# -----------------------------------------------------------------------------
# Core Modules
# -----------------------------------------------------------------------------

class MultiScaleIntegrator(nn.Module):
    """
    多尺度特征积分器。
    使用卷积层对不同尺度的特征进行加权融合。
    """

    def __init__(self, num_scales: int = 3):
        super().__init__()
        # 使用 1xK 卷积在 scales 维度上进行融合
        # Input: [Batch, 1, Hidden, Scales] -> Output: [Batch, 1, Hidden, 1]
        self.integrating_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, num_scales),
            stride=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape [batch, hidden_dim, num_scales]
        Returns:
            shape [batch, hidden_dim]
        """
        # [batch, hidden, scales] -> [batch, 1, hidden, scales]
        x = x.unsqueeze(1)
        x = self.integrating_conv(x)
        # [batch, 1, hidden, 1] -> [batch, hidden]
        x = x.squeeze(3).squeeze(1)
        return x


class MultiScaleFusion(nn.Module):
    """
    多尺度融合模块 (Mixture of Scales)。
    """

    def __init__(self, input_dim: int, output_dim: int, config: MultiModalConfig):
        super().__init__()

        self.hidden_dim = config.fusion_hidden_dim

        # 动态构建多尺度专家层
        self.experts = nn.ModuleList()
        for factor in config.scale_factors:
            expert = nn.Sequential(
                nn.Linear(input_dim, output_dim // factor),
                nn.GELU(),
                nn.Linear(output_dim // factor, self.hidden_dim)
            )
            self.experts.append(expert)

        assert len(self.experts) == config.num_fusion_scales, \
            "Defined scale factors count must match num_fusion_scales"

        self.integrator = MultiScaleIntegrator(num_scales=config.num_fusion_scales)
        self.projector = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape [batch, input_dim]
        """
        # 处理单个样本输入的情况
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 计算所有专家的输出
        # List of [batch, hidden_dim]
        expert_outputs = [expert(x) for expert in self.experts]

        # Stack logic: [batch, hidden_dim, num_scales]
        multi_scale_stack = torch.stack(expert_outputs, dim=2)

        # 积分融合: [batch, hidden_dim]
        integrated_feat = self.integrator(multi_scale_stack)

        # 最终投影: [batch, output_dim]
        return self.projector(integrated_feat)


class AudioVideoProjector(nn.Module):
    """
    音视频特征投影与融合模块 (原 MultiModal_MSE)。
    负责将原始音视频特征投影到统一空间并相加，随后通过多尺度融合。
    """

    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self.config = config

        # 模态对齐层
        self.video_proj = nn.Linear(config.video_feat_dim, config.fusion_hidden_dim)
        self.audio_proj = nn.Linear(config.audio_feat_dim, config.fusion_hidden_dim)

        # 多尺度融合器
        self.fusion_module = MultiScaleFusion(
            input_dim=config.fusion_hidden_dim,
            output_dim=config.output_embed_dim,
            config=config
        )

    def forward(self, video_feat: torch.Tensor, audio_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_feat: [batch, seq_len_v, v_dim]
            audio_feat: [batch, seq_len_a, a_dim]
        Returns:
            [batch, output_embed_dim]
        """
        # 1. 全局平均池化 (Temporal Pooling)
        # [batch, seq_len, dim] -> [batch, 1, dim]
        video_avg = torch.mean(video_feat, dim=1, keepdim=True)
        audio_avg = torch.mean(audio_feat, dim=1, keepdim=True)

        # 2. 投影到统一维度
        video_embeds = self.video_proj(video_avg)
        audio_embeds = self.audio_proj(audio_avg)

        # 3. 简单的加法融合
        # [batch, 1, hidden_dim]
        fused_input = video_embeds + audio_embeds

        # 移除 seq_len=1 的维度 -> [batch, hidden_dim]
        fused_input = fused_input.squeeze(1)

        # 4. 通过多尺度融合模块
        return self.fusion_module(fused_input)


# -----------------------------------------------------------------------------
# Main Model Wrapper
# -----------------------------------------------------------------------------

class ChatGLMForMultimodal(nn.Module):
    def __init__(self, LLM_model: nn.Module, config: Optional[MultiModalConfig] = None):
        super().__init__()
        self.LLM_model = LLM_model
        self.config = config if config is not None else MultiModalConfig()

        # 核心多模态投影模块
        self.mm_projector = AudioVideoProjector(self.config)

    def _process_single_sample_features(self, raw_features: List[torch.Tensor]) -> torch.Tensor:
        """
        处理单个样本中的多模态特征列表。
        假设列表结构为 [Video1, Audio1, Video2, Audio2, ...]
        """
        if len(raw_features) % 2 != 0:
            raise ValueError(f"Raw features length must be even (pairs of video/audio), got {len(raw_features)}")

        fused_tokens = []

        # 步长为 2 遍历
        for i in range(0, len(raw_features), 2):
            video_tensor = raw_features[i]  # [seq_len, v_dim]
            audio_tensor = raw_features[i + 1]  # [seq_len, a_dim]

            # 添加 batch 维度以适应 Module forward 接口
            # [1, seq_len, dim]
            video_input = video_tensor.unsqueeze(0)
            audio_input = audio_tensor.unsqueeze(0)

            # 前向传播 (Feature Projection)
            # output: [1, output_dim]
            projected_feat = self.mm_projector(video_input, audio_input)
            fused_tokens.append(projected_feat)

        if not fused_tokens:
            raise ValueError("Processed sample resulted in no tokens.")

        # 将列表堆叠为 Tensor: [num_pairs, output_dim]
        return torch.cat(fused_tokens, dim=0)

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: torch.Tensor,
            raw_multimodal_features: List[List[torch.Tensor]],
            modality_types: List[str], 
            multimodal_indices: torch.Tensor,
            **kwargs
    ):
        """
        Args:
            raw_multimodal_features: Batch List of Sample Lists containing tensors.
                                     [[v1, a1], [v1, a1, v2, a2], ...]
        """
        batch_multimodal_embeds = []

        if raw_multimodal_features:
            # 1. 处理每个样本的特征
            for i, sample_features in enumerate(raw_multimodal_features):
                # 获得该样本所有的多模态 embedding: [num_tokens, embed_dim]
                sample_embeds = self._process_single_sample_features(sample_features)

                # 校验生成的 token 数量是否与索引一致
                expected_count = len(multimodal_indices[i])
                if sample_embeds.shape[0] != expected_count:
                    raise AssertionError(
                        f"Sample {i}: Token mismatch. Generated {sample_embeds.shape[0]}, "
                        f"Expected {expected_count} (from indices)."
                    )

                batch_multimodal_embeds.append(sample_embeds)

            # 2. 处理 Batch Padding (Left Padding)
            # 计算 Batch 中最大的 token 数量
            max_tokens = max(t.size(0) for t in batch_multimodal_embeds)

            padded_batch = []
            embed_dim = batch_multimodal_embeds[0].size(1)
            dtype = batch_multimodal_embeds[0].dtype
            device = batch_multimodal_embeds[0].device

            for embeds in batch_multimodal_embeds:
                curr_len = embeds.size(0)
                pad_len = max_tokens - curr_len

                if pad_len > 0:
                    # 创建全0 padding
                    padding = torch.zeros((pad_len, embed_dim), dtype=dtype, device=device)
                    # Left padding: [PAD, Embeds]
                    embeds = torch.cat([padding, embeds], dim=0)

                padded_batch.append(embeds)

            # [Batch, Max_Tokens, Embed_Dim]
            final_multimodal_embeds = torch.stack(padded_batch)

        else:
            # 处理无多模态输入的 Corner Case
            raise ValueError("No multimodal features provided.")

        # 3. 调用底层 LLM_model
        final_inputs = {
            "input_ids": input_ids,
            "labels": labels,
            "multimodal_embeddings": final_multimodal_embeds,
            "multimodal_indices": multimodal_indices,
            **kwargs
        }

        return self.LLM_model(**final_inputs)
