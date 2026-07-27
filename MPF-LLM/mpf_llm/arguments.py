from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """与模型 / config / tokenizer 相关的参数。"""

    model_name_or_path: str = field(
        default="THUDM/chatglm3-6b-base",
        metadata={"help": "预训练模型路径或 huggingface 模型标识"},
    )
    lora_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "LoRA checkpoint 路径"}
    )
    config_name: Optional[str] = field(default=None, metadata={"help": "自定义 config 名称/路径"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "自定义 tokenizer 名称/路径"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "预训练模型缓存目录"})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "是否使用 fast tokenizer"})
    model_revision: str = field(default="main", metadata={"help": "模型版本(branch/tag/commit)"})
    use_auth_token: bool = field(default=False, metadata={"help": "是否使用 hf 登录 token(私有模型)"})

    quantization_bit: Optional[int] = field(
        default=None, metadata={"help": "量化位数；None 表示不量化"}
    )

    # ---- LoRA 超参 ----
    lora_rank: int = field(default=8, metadata={"help": "LoRA rank，越大表达力越强但参数量更多"})
    lora_alpha: float = field(default=32, metadata={"help": "LoRA alpha 缩放系数"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})


@dataclass
class DataTrainingArguments:
    """与训练/评估数据相关的参数。"""

    train_file: str = field(
        default="./data/meca_train.json",
        metadata={"help": "训练数据文件(json / jsonl)"},
    )
    val_file: Optional[str] = field(default=None, metadata={"help": "验证数据文件(json / jsonl)"})

    # 多模态预抽取特征（.pt 字典：{key: tensor}）
    video_features_path: str = field(
        default="./data/features/video_features.pt",
        metadata={"help": "视频特征 .pt 路径"},
    )
    audio_features_path: str = field(
        default="./data/features/audio_features.pt",
        metadata={"help": "音频特征 .pt 路径"},
    )

    # 真正生效的长度控制是下面两个；序列总长 = max_source_length + max_target_length + 1(eos)
    max_source_length: int = field(default=1100, metadata={"help": "source 端最大长度，超出截断"})
    max_target_length: int = field(default=128, metadata={"help": "target 端最大长度，超出截断"})
    # 仅为与启动脚本兼容而保留；实际序列长度由 source/target 决定，此项不参与截断
    max_seq_length: Optional[int] = field(default=None, metadata={"help": "(兼容项, 不参与实际截断)"})

    train_format: Optional[str] = field(
        default="input-output", metadata={"help": "训练数据格式: multi-turn 或 input-output"}
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "覆盖缓存"})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "预处理进程数"})
    pad_to_max_length: bool = field(default=False, metadata={"help": "是否填充到固定最大长度"})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "截断训练样本数(调试用)"})

    def __post_init__(self):
        extension = self.train_file.split(".")[-1]
        if extension not in {"json", "jsonl"}:
            raise ValueError("`train_file` 应为 json 或 jsonl 文件。")
        if self.train_format not in {"multi-turn", "input-output"}:
            raise ValueError("`train_format` 应为 'multi-turn' 或 'input-output'。")
