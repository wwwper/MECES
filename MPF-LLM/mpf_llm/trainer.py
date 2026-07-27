# coding=utf-8
"""
自定义 Trainer。

    - compute_loss 增加 **kwargs，兼容新版 transformers 传入的 num_items_in_batch 等参数。
    - save_model 分别保存 LoRA 适配器与融合模块权重。
"""

import os
from typing import Optional

import torch
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.utils import logging

logger = logging.get_logger(__name__)

TRAINING_ARGS_NAME = "training_args.bin"
FUSION_WEIGHTS_NAME = "fusion_module.pt"


class LoRATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        model_to_save = unwrap_model(self.model)

        # 1) LoRA 适配器（backbone 已被 PEFT 包裹）
        model_to_save.backbone.save_pretrained(output_dir)
        logger.info(f"LoRA adapters saved to {output_dir}")

        # 2) 融合模块权重
        fusion_path = os.path.join(output_dir, FUSION_WEIGHTS_NAME)
        torch.save(model_to_save.fusion_model.state_dict(), fusion_path)
        logger.info(f"Fusion module saved to {fusion_path}")

        # 3) tokenizer 与训练参数
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
