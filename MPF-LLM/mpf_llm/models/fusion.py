"""
多模态多层级融合模块 (MultiModal_MLF)。

把一对 (video, audio) 预抽取特征融合成单个伪 token 向量，供注入到 LLM 的 word embedding 序列中。
"""

import torch
import torch.nn as nn


class Integrating(nn.Module):

    def __init__(self, scales: int):
        super().__init__()
        self.integrating_layer = nn.Conv2d(1, 1, kernel_size=(1, scales), stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, hidden, scales]
        x = x.unsqueeze(1)              # [batch, 1, hidden, scales]
        x = self.integrating_layer(x)  # [batch, 1, hidden, 1]
        x = x.squeeze(-1).squeeze(1)   # [batch, hidden]
        return x


class MultiScaleFusion(nn.Module):

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 768):
        super().__init__()

        def expert(bottleneck: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_size, bottleneck),
                nn.GELU(),
                nn.Linear(bottleneck, hidden_size),
            )

        self.scale1 = expert(output_size // 8)
        self.scale2 = expert(output_size // 32)
        self.scale3 = expert(output_size // 16)

        self.integrating = Integrating(scales=3)
        self.projector = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, input_size]
        if x.dim() == 1:                       # 兼容单样本输入
            x = x.unsqueeze(0)

        stack = torch.stack(                   # [batch, hidden, 3]
            [self.scale1(x), self.scale2(x), self.scale3(x)], dim=2
        )
        integrated = self.integrating(stack)   # [batch, hidden]
        return self.projector(integrated)      # [batch, output_size]


class MultiModal_MLF(nn.Module):
    """
    把一对 (video, audio) 特征融合成单个伪 token 向量。

    video_hidden_state: [batch, video_seq_len, video_feat_dim]
    audio_hidden_state: [batch, audio_seq_len, audio_feat_dim]
    return            : [batch, fusion_output_size]
    """

    def __init__(self, fusion_input_size: int, fusion_output_size: int,
                 video_feat_dim: int, audio_feat_dim: int, hidden_size: int = 768):
        super().__init__()
        self.video_proj = nn.Linear(video_feat_dim, hidden_size)
        self.audio_proj = nn.Linear(audio_feat_dim, hidden_size)
        self.fusion = MultiScaleFusion(
            input_size=fusion_input_size, output_size=fusion_output_size, hidden_size=hidden_size
        )

    def forward(self, video_hidden_state: torch.Tensor, audio_hidden_state: torch.Tensor) -> torch.Tensor:
        # 使权重与输入 dtype 对齐（fp16 训练时 .pt 特征常为 fp32，避免 dtype mismatch）
        weight_dtype = self.video_proj.weight.dtype
        video_hidden_state = video_hidden_state.to(weight_dtype)
        audio_hidden_state = audio_hidden_state.to(weight_dtype)

        # 时间维平均池化 -> [batch, 1, feat_dim]
        video_avg = video_hidden_state.mean(dim=1, keepdim=True)
        audio_avg = audio_hidden_state.mean(dim=1, keepdim=True)

        # 投影到统一维度并相加 -> [batch, 1, hidden] -> [batch, hidden]
        x = (self.video_proj(video_avg) + self.audio_proj(audio_avg)).squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"融合输入维度异常，期望 [batch, hidden]，实际 {tuple(x.shape)}")

        return self.fusion(x)  # [batch, fusion_output_size]
