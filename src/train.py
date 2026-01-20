import logging
import os
import sys
import json
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import transformers
from transformers import (
    Trainer,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    PreTrainedTokenizer
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType
)

# 本地模块导入
from trainer import LoRATrainer
from arguments import ModelArguments, DataTrainingArguments
from preprocess_utils import sanity_check, InputOutputDataset
from datacollator import MultimodalDataCollator
from model import ChatGLMForMultimodal

# 设置日志格式
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """设置所有相关的随机种子以保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CastOutputToFloat(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs).float()

def load_dataset(file_path: str, format_type: str, tokenizer: PreTrainedTokenizer, data_args: DataTrainingArguments) -> InputOutputDataset:
    """加载并处理数据集的辅助函数"""
    if not file_path:
        return None
        
    logger.info(f"Loading data from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".json"):
            data = json.load(f)
        elif file_path.endswith(".jsonl"):
            data = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    if format_type == "input-output":
        dataset = InputOutputDataset(
            data,
            tokenizer,
            data_args.max_source_length,
            data_args.max_target_length,
        )
    else:
        raise ValueError(f"Unknown train format: {format_type}")
    
    return dataset

def main():
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 初始化日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")

    set_seed(training_args.seed) # 使用参数中的 seed

    # 加载 Tokenizer 并添加特殊 Token
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    
    VIDEO_PLACEHOLDER = "<video_placeholder>"
    AUDIO_PLACEHOLDER = "<audio_placeholder>"
    special_tokens_dict = {'additional_special_tokens': [VIDEO_PLACEHOLDER, AUDIO_PLACEHOLDER]}
    tokenizer.add_special_tokens(special_tokens_dict)

    #  加载模型
    logger.info("Loading base model...")
    # 注意：在多卡训练时，通常由 Trainer 或 Accelerator 处理 .cuda()，但在特定自定义模型中可能需要手动处理
    base_model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True).cuda()
    
    # 包装为多模态模型
    model = ChatGLMForMultimodal(base_model)

    if model_args.quantization_bit is not None:
        logger.info(f"Quantizing model to {model_args.quantization_bit} bit")
        model = base_model.quantize(model_args.quantization_bit)

    #PEFT / LoRA 配置
    logger.info("Applying PEFT (LoRA) configuration...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_rank,
        target_modules=['query_key_value'],
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
    )
    # 仅对 Transformer 部分应用 LoRA，并确保移动到 GPU
    model.transformer = get_peft_model(model.transformer, peft_config).to("cuda")
    
    # 打印可训练参数情况
    model.transformer.print_trainable_parameters()

    # 6. 模型训练特定设置 (Gradient Checkpointing & Precision)
    model.transformer.enable_input_require_grads()
    model.transformer.is_parallelizable = True
    model.transformer.model_parallel = True
    # 确保输出层精度正确
    model.transformer.lm_head = CastOutputToFloat(model.transformer.transformer.output_layer)
    model.transformer.config.use_cache = False

    # 加载数据集
    logger.info("Loading training dataset...")
    train_dataset = load_dataset(data_args.train_file, data_args.train_format, tokenizer, data_args)
    print(f"Train dataset size: {len(train_dataset)}")

    # 验证集
    val_dataset = None
    if training_args.do_eval and data_args.val_file:
        logger.info("Loading validation dataset...")
        val_dataset = load_dataset(data_args.val_file, data_args.train_format, tokenizer, data_args)
        print(f"Validation dataset size: {len(val_dataset)}")

    # 数据集完整性检查 (Sanity Check)
    if train_dataset:
        sanity_check(train_dataset[0]['input_ids'], train_dataset[0]['labels'], tokenizer)

    #  Data Collator
    data_collator = MultimodalDataCollator(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False
    )


    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, 
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    if training_args.resume_from_checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {training_args.resume_from_checkpoint}")
    
    # 二次确保梯度设置（有些模型在 Trainer init 后会重置）
    model.transformer.enable_input_require_grads()

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # 保存模型
    logger.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()

if __name__ == "__main__":
    main()
