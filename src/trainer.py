import os
import torch
from typing import Dict, Union, Any, Optional, Tuple, List

from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.utils import logging

# 设置日志
logger = logging.get_logger(__name__)

TRAINING_ARGS_NAME = "training_args.bin"
MMProjector_NAME = "mm_projector.pt"

class LoRATrainer(Trainer):
    """
    自定义 Trainer，用于处理多模态模型的 LoRA 训练。
    主要重写了 compute_loss 和 save_model 方法，以支持特定组件（LoRA + mm_projector）的保存。
    """

    def compute_loss(
        self, 
        model: Any, 
        inputs: Dict[str, Union[torch.Tensor, Any]], 
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        重写 compute_loss 以适应多模态模型的输出格式。
        """
        # 前向传播
        outputs = model(**inputs)

        # 获取 Loss
        loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving model checkpoint to {output_dir}")

        # 获取解包后的模型（去除 DDP/DataParallel 包装）
        model_to_save = unwrap_model(self.model)


        # 假设 model.transformer 是挂载了 LoRA 的 PeftModel
        if hasattr(model_to_save, "transformer"):
            # PEFT 的 save_pretrained 会自动只保存 LoRA 权重
            model_to_save.transformer.save_pretrained(output_dir)
            logger.info(f"LoRA adapters saved to {output_dir}")
        else:
            logger.warning("Attribute 'transformer' not found in model, skipping LoRA save.")

        if hasattr(model_to_save, "mm_projector"):
            mm_projector_state_dict = model_to_save.mm_projector.state_dict()
            save_path = os.path.join(output_dir, MMProjector_NAME)
            torch.save(mm_projector_state_dict, save_path)
            logger.info(f"mm_projector_model model saved to {save_path}")
        else:
            logger.warning("Attribute 'mm_projector_model' not found in model")


        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
