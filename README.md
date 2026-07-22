<div align="center">

# Locate and Explain: Joint Multimodal Emotion Cause Extraction and Summarization in Conversation

[![Paper](https://img.shields.io/badge/Paper-ACL%202026-b31b1b.svg)](https://aclanthology.org/2026.acl-long.2012/)

<img src="https://github.com/user-attachments/assets/888151d8-8ce8-42db-b481-28e81c36300e" width="720" alt="Overview of MPF-LLM">

📄 [**Paper**](https://aclanthology.org/2026.acl-long.2012/) &nbsp;|&nbsp; 📊 [**Dataset (MECESD)**](#-数据集-mecesd) &nbsp;|&nbsp; 🚀 [**Quick Start**](#-安装) &nbsp;|&nbsp; 📝 [**Citation**](#-引用-citation)

</div>

---

---

本仓库是论文 **《Locate and Explain: Joint Multimodal Emotion Cause Extraction and Summarization in Conversation》**（ACL 2026, Long Papers）的官方实现，包含**MECESD 数据集** 以及 **MPF-LLM 模型** 的完整训练与推理代码。

> **作者**：Jikun Wan, Chen Gong\*, Guohong Fu &nbsp;(\* 通讯作者)
> **单位**：苏州大学 人工智能研究院 / 计算机科学与技术学院


## 📢 News

- **2026-07** 论文被 **ACL 2026** 接收（Volume 1: Long Papers, pp. 43472–43489）。
- 数据集与代码整理发布中。


## 📊 数据集 MECESD

**MECESD** 是首个**同时**为情绪原因**抽取**与**总结**标注的多模态对话数据集。其标注以心理学中的 **ABC 理论**（Activating events–Beliefs–Consequences，Ellis, 1957）为指导，从「激发事件」与「信念」双重视角刻画情绪成因，从而提升标注的全面性与一致性。

### 获取数据集

> ⚠️ **使用许可**：MECESD 仅限**非商业学术研究**用途，禁止任何商业或非学术使用。所有标注基于公开 M3ED 数据集并已获得原始数据所有者授权。请在下载前阅读并同意相应许可条款。

数据集将按如下方式提供（请根据实际发布情况替换链接）：

```bash
# 下载标注 json 与预抽取多模态特征，放入 data/ 目录
# <此处填写你的数据集下载链接，例如 Google Drive / 百度网盘 / HuggingFace Datasets>
```

最终目录组织见下方[数据准备](#-数据准备)。


## 📁 目录结构

```
MPF-LLM/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── train_meca.sh          # 训练启动脚本
│   └── infer_meca.sh          # 推理启动脚本
└── mpf_llm/                    # 主包
    ├── arguments.py           # ModelArguments / DataTrainingArguments
    ├── trainer.py             # LoRATrainer（自定义 compute_loss 与 save_model）
    ├── train.py               # 训练入口（python -m mpf_llm.train）
    ├── inference.py           # 推理入口（python -m mpf_llm.inference）
    ├── data/
    │   ├── dataset.py         # InputOutputDataset + 特征加载
    │   └── collator.py        # MultimodalDataCollator
    └── models/
        ├── fusion.py          # Integrating / MultiScaleFusion / MultiModal_MLF
        └── modeling_mpf_llm.py# MPF_LLM 复合模型
```

## ⚙️ 安装

```bash
git clone <your-repo-url> && cd MPF-LLM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> ChatGLM3-6B 对 `transformers` 版本较敏感，若加载报错请优先调整该版本。

## 🗂️ 数据准备

将数据放置于 `data/`（已在 `.gitignore` 中，不入库）：

1. **训练/测试 json**：每条样本至少包含
   - `context`：含 `<video_placeholder>` / `<audio_placeholder>` 占位符的输入文本；
   - `target`：期望输出文本（原因话语索引 + 原因概括）；
   - `multimodal_features_key_list`：占位符对应的特征 key 列表（顺序与占位符一致，one-token 版本每个 key 对应 1 个占位符）。
2. **多模态特征 `.pt`**：`video_features.pt` / `audio_features.pt`，均为 `{key: tensor}` 字典，`key` 与上面的 `multimodal_features_key_list` 对应。视觉/音频特征分别由 **CLIP ViT-L** 与 **HuBERT-L** 预抽取。

```
data/
├── meca_train.json
├── meca_test.json
├── meca_test_gold.json
└── features/
    ├── video_features.pt
    └── audio_features.pt
```

## 🚀 训练

```bash
bash scripts/train_meca.sh
```

或直接调用（从仓库根目录）：

```bash
torchrun --nproc_per_node=2 -m mpf_llm.train \
    --train_format input-output \
    --train_file ./data/meca_train.json \
    --video_features_path ./data/features/video_features.pt \
    --audio_features_path ./data/features/audio_features.pt \
    --model_name_or_path THUDM/chatglm3-6b-base \
    --output_dir ./checkpoints/meca_mlf \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --remove_unused_columns False \
    --fp16 --gradient_checkpointing
```

产物保存在 `output_dir`：LoRA 适配器、`fusion_module.pt`、tokenizer、训练参数。

**默认超参**（对应论文 Appendix A.10）：Epochs = 3，Learning Rate = 1e-4，Per-device Batch Size = 1，Gradient Accumulation = 4（有效 batch = 8），LoRA `r = 8`、`α = 32`、`dropout = 0.1`；默认上下文窗口 window(8, 3)。实验在 32GB V100 / 40GB A100 上完成。

## 🔮 推理与评估

```bash
bash scripts/infer_meca.sh
```

或：

```bash
python -m mpf_llm.inference \
    --model THUDM/chatglm3-6b-base \
    --lora_path ./checkpoints/meca_mlf/checkpoint-1581 \
    --video_features_path ./data/features/video_features.pt \
    --audio_features_path ./data/features/audio_features.pt \
    --ecp_test_path ./data/meca_test.json \
    --gold_emo_path ./data/meca_test_gold.json \
    --save_pred_path ./results/meca_mlf_pred.json
```

**评估指标**：MECE 子任务采用加权平均 **F1**；MECS 子任务采用 **BLEU-2 / BLEU-4 / METEOR / ROUGE-L**（词面重叠）以及 **BERTScore / Sentence-BERT**（语义相似度）。
## 💡 说明与注意事项

- **backbone 要求**：`MPF_LLM.forward` 依赖一个被改造过、可接收 `multimodal_embeddings` 与 `multimodal_indices` 的 LLM。若更换 backbone，需要相应改造其 embedding 注入逻辑。
- **batch 语义**：当前对 `per_device_train_batch_size=1` 完全正确；大于 1 且各样本多模态 token 数不同时，需要 backbone 忽略 `index == -1` 的填充位。
- **权重文件命名**：融合模块保存为 `fusion_module.pt`（旧版本名为 `qformer.pt`，如复用旧 checkpoint 重命名即可）。

## 📝 引用 Citation

如果本仓库或数据集对你的研究有帮助，请引用：

```bibtex
@inproceedings{wan-etal-2026-locate,
    title     = "Locate and Explain: Joint Multimodal Emotion Cause Extraction and Summarization in Conversation",
    author    = "Wan, Jikun and Gong, Chen and Fu, Guohong",
    booktitle = "Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month     = jul,
    year      = "2026",
    publisher = "Association for Computational Linguistics",
    pages     = "43472--43489",
    url       = "https://aclanthology.org/2026.acl-long.2012/"
}
```

> 官方 BibTeX（含出版地址等完整字段）可从 [ACL Anthology 页面](https://aclanthology.org/2026.acl-long.2012/) 复制。

## 🙏 致谢 Acknowledgements

- 数据集基于 **M3ED**（Zhao et al., 2022）构建。


## 📄 License

代码采用 **MIT** 许可，见 [LICENSE](./LICENSE)。**数据集仅限非商业学术研究用途**，详见[数据集许可说明](#获取数据集)。
