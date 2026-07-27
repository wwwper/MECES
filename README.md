<div align="center">

# Locate and Explain: Joint Multimodal Emotion Cause Extraction and Summarization in Conversation

[![Paper](https://img.shields.io/badge/Paper-ACL%202026-b31b1b.svg)](https://aclanthology.org/2026.acl-long.2012/)

<img src="https://github.com/user-attachments/assets/888151d8-8ce8-42db-b481-28e81c36300e" width="720" alt="Overview of MPF-LLM">

📄 [**Paper**](https://aclanthology.org/2026.acl-long.2012/) &nbsp;|&nbsp; 📊 [**Dataset (MECESD)**](#-数据集-mecesd) &nbsp;|&nbsp; 🚀 [**Quick Start**](#-安装) &nbsp;|&nbsp; 📝 [**Citation**](#-引用-citation)

</div>

---


本仓库是论文 **《Locate and Explain: Joint Multimodal Emotion Cause Extraction and Summarization in Conversation》**（ACL 2026, Long Papers）的官方实现，包含**MECESD 数据集** 以及 **MPF-LLM 模型** 的完整训练与推理代码。

> **作者**：Jikun Wan, Chen Gong\*, Guohong Fu &nbsp;(\* 通讯作者)
> **单位**：苏州大学 人工智能研究院 / 计算机科学与技术学院

## 📊 数据集 MECESD

**MECESD** 是首个**同时**为情绪原因**抽取**与**总结**标注的多模态对话数据集。其标注以心理学中的 **ABC 理论**（Activating events–Beliefs–Consequences，Ellis, 1957）为指导，从「激发事件」与「信念」双重视角刻画情绪成因，从而提升标注的全面性与一致性。

### 获取数据集

> ⚠️ **使用许可**：MECESD 仅限**非商业学术研究**用途，禁止任何商业或非学术使用。

# 如果需要下载数据集对应的视频，或者直接使用提取的视频、音频特征放入 dataset/multimodal_features文件路径，请通过邮件联系我们，并填写相应的许可条款。


## ⚙️ 安装

```bash
git clone <your-repo-url> && cd MPF-LLM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> 环境中还有一些其它库需要安装，详细请参考代码

## 🗂️ 数据准备

将数据放置于 `data/`（已在 `.gitignore` 中，不入库）：

1. **训练/验证/测试 json**：每条样本至少包含
   - `context`；
   - `target`；
   - `multimodal_features_key_list`：占位符对应的特征 key 列表（顺序与占位符一致）。
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
**默认超参**（对应论文 Appendix A.10）：Epochs = 3，Learning Rate = 1e-4，Per-device Batch Size = 1，Gradient Accumulation = 4，LoRA `r = 8`、`α = 32`、`dropout = 0.1`；默认上下文窗口 window(8, 3)。实验在 32GB V100 / 40GB A100 上完成。

## 推理与评估

```bash
bash scripts/infer_meca.sh
```

**评估指标**：MECE 子任务采用加权平均 **F1**；MECS 子任务采用 **BLEU-2 / BLEU-4 / METEOR / ROUGE-L**（词面重叠）以及 **BERTScore / Sentence-BERT**（语义相似度）。
其中词面重叠指标使用 [nlg-eval](https://github.com/Maluuba/nlg-eval) 计算 BLEU、METEOR、ROUGE-L等指标。
BERTScore使用bert-base-chinese模型(中文语料中评估表现最好的，详细参考https://github.com/Tiiiger/bert_score)，Sentence-BERT使用paraphrase-multilingual-MiniLM-L12-v2。详细请参考代码
## 💡 说明与注意事项

- **backbone 要求**：若更换 backbone，需要相应改造其 embedding 注入/替换逻辑，由于chatglm的模型的generate方法不支持embedding输入，所以我们修改了对应的modeling_chatglm.py文件，对于其它LLM backbone可以直接通过传入inputs_embeds进行Embedding替换。


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

- 数据集基于 **M3ED**（Zhao et al., 2022）构建，所有标注基于公开 M3ED 数据集并已获得原始数据所有者授权。


## 📄 License

代码采用 **MIT** 许可，见 [LICENSE](./LICENSE)。**数据集仅限非商业学术研究用途**，详见[数据集许可说明](#获取数据集)。
