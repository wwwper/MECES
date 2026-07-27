"""
多模态 LoRA 微调入口。

运行 (从仓库根目录):
    torchrun --nproc_per_node=2 -m mpf_llm.train --train_file ... --output_dir ...
或参见 scripts/train_meca.sh
"""

import json
import logging
import os
import random
import sys

import numpy as np
import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
)

from mpf_llm.arguments import DataTrainingArguments, ModelArguments
from mpf_llm.data import (
    MULTIMODAL_PLACEHOLDER,
    InputOutputDataset,
    MultimodalDataCollator,
    load_multimodal_features,
    sanity_check,
)
from mpf_llm.models import MPF_LLM
from mpf_llm.trainer import LoRATrainer

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CastOutputToFloat(torch.nn.Module):
    """在 fp16 训练时把输出层结果转 float，提升 loss 数值稳定性。"""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs).float()


def load_json_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f]
        return json.load(f)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(42)

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
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed: {training_args.local_rank != -1}, "
        f"fp16: {training_args.fp16}"
    )

    # ---- tokenizer + 特殊 token ----
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [MULTIMODAL_PLACEHOLDER]}
    )

    # ---- base model ----
    base_model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    # 词表随特殊 token 扩展（fp32 加载，混合精度由 Trainer AMP 负责）

    # ChatGLM3 的 padded_vocab_size (65024) 通常大于 tokenizer 实际词表，新增的占位符 token
    # 会落在空余行中，无需 resize；且 ChatGLM 未实现 set_input_embeddings，强行 resize 会报
    # NotImplementedError。因此仅在词表确实超出 embedding 行数时才扩展。
    embed_rows = base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embed_rows:
        logger.info(f"Resizing token embeddings: {embed_rows} -> {len(tokenizer)}")
        base_model.resize_token_embeddings(len(tokenizer))
    else:
        logger.info(f"跳过 resize：embedding 行数 {embed_rows} >= 词表大小 {len(tokenizer)}")

    if model_args.quantization_bit is not None:
        logger.info(f"Quantizing to {model_args.quantization_bit} bit")
        base_model = base_model.quantize(model_args.quantization_bit)

    base_model = base_model.cuda()
    model = MPF_LLM(base_model)

    # ---- 数据 ----
    image_features, audio_features = load_multimodal_features(
        data_args.video_features_path, data_args.audio_features_path
    )
    train_data = load_json_data(data_args.train_file)

    if data_args.train_format != "input-output":
        raise ValueError(f"Unsupported train format: {data_args.train_format}")

    train_dataset = InputOutputDataset(
        train_data, tokenizer, image_features, audio_features,
        data_args.max_source_length, data_args.max_target_length,
    )
    logger.info(f"Train dataset size: {len(train_dataset)}")
    sanity_check(train_dataset[0]["input_ids"], train_dataset[0]["labels"], tokenizer)

    # ---- LoRA ----
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_rank,
        target_modules=["query_key_value"],
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
    )
    model.backbone = get_peft_model(model.backbone, peft_config).to("cuda")

    # 梯度检查点 / require grads 所需设置（不设置与 DDP 冲突的 model_parallel）
    model.backbone.enable_input_require_grads()
    # 注意: model.backbone 是完整 LLM，其内部 .transformer 才是 ChatGLM 主干，.output_layer 是输出头
    model.backbone.lm_head = CastOutputToFloat(model.backbone.transformer.output_layer)
    model.backbone.config.use_cache = False

    data_collator = MultimodalDataCollator(tokenizer=tokenizer, label_pad_token_id=-100)

    trainer = LoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
