<div align="center">

# Locate and Explain: Joint Multimodal Emotion Cause Extraction and Summarization in Conversation

[![Paper](https://img.shields.io/badge/Paper-ACL%202026-b31b1b.svg)](https://aclanthology.org/2026.acl-long.2012/)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](./LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-MECESD-blue.svg)](#-dataset-mecesd)

<img src="https://github.com/user-attachments/assets/888151d8-8ce8-42db-b481-28e81c36300e" width="720" alt="Overview of MPF-LLM">

📄 [**Paper**](https://aclanthology.org/2026.acl-long.2012/) &nbsp; &nbsp; 📊 [**Dataset (MECESD)**](#-dataset-mecesd) &nbsp; &nbsp; 🚀 [**Quick Start**](#-installation) &nbsp; &nbsp; 📝 [**Citation**](#-citation)

</div>

---

This repository is the **official implementation** of the paper **"Locate and Explain: Joint Multimodal Emotion Cause Extraction and Summarization in Conversation"** (ACL 2026, Long Papers). It provides the **MECESD dataset** together with the complete training and inference code for the **MPF-LLM model**.

---

## 📊 Dataset: MECESD

**MECESD** is the **first** multimodal conversational dataset annotated for **both** emotion-cause **extraction** and **summarization**.

Its annotation is guided by the psychological **ABC theory**  — which characterizes the causes of an emotion from the dual perspectives of *activating events* and *beliefs*. This dual view improves both the **comprehensiveness** and the **consistency** of the annotation.

### Getting the Dataset

> ⚠️ **License:** MECESD is restricted to **non-commercial academic research** only. Any commercial or non-academic use is strictly prohibited.

To download the videos associated with the dataset, or to directly use the pre-extracted video/audio features, please **contact us by email** and sign the corresponding license agreement.

---

## 🔧 Installation

```bash
git clone <your-repo-url> && cd MPF-LLM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> ℹ️ A few additional libraries may be required beyond `requirements.txt`. Please refer to the source code for the full list.

---

## 📂 Data Preparation

Place all data under the `dataset/` directory.

**1. Build the fine-tuning data** (train / val / test) via `data/bulid_finetune_dataset.py`. Each sample contains at least:

- `context`
- `target`
- `multimodal_features_key_list` — the list of feature keys corresponding to the placeholders, **in the same order** as the placeholders.

**2. Prepare the downloaded multimodal features** (`.pt` files): `video_features.pt` and `audio_features.pt`. Each is a `{key: tensor}` dictionary whose `key` matches the `multimodal_features_key_list` above. Visual and audio features are pre-extracted with **CLIP ViT-L** and **HuBERT-L**, respectively.

**Expected directory layout:**

```text
dataset/
├── MECESD_train.json
├── MECESD_val.json
├── MECESD_test.json
├── multimodal_features/
│   ├── video_features.pt
│   └── audio_features.pt
└── finetune_prompt_data/
    ├── MECES_multimodal_train_finetune.json
    ├── MECES_multimodal_val_finetune.json
    └── MECES_multimodal_test_finetune.json
```

---

## 🚀 Training

```bash
bash scripts/train_meca.sh
```

**Default hyperparameters** (corresponding to Appendix A.10 of the paper):

| Setting | Value |
| --- | --- |
| Epochs | 3 |
| Learning rate | 1e-4 |
| Per-device batch size | 1 |
| Gradient accumulation | 4 |
| LoRA `r` / `α` / `dropout` | 8 / 32 / 0.1 |
| Context window | `window(8, 3)` |

> Experiments were conducted on **32GB V100** / **40GB A100** GPUs.

---

## 📈 Inference & Evaluation

```bash
bash scripts/infer_meca.sh
```

**Evaluation metrics:**

- **MECE** subtask — weighted **F1**.
- **MECS** subtask — **BLEU-2 / BLEU-4 / METEOR / ROUGE-L** (surface-level lexical overlap) and **BERTScore / Sentence-BERT** (semantic similarity).

**Implementation details:**

- Lexical-overlap metrics (BLEU, METEOR, ROUGE-L, etc.) are computed with [nlg-eval](https://github.com/Maluuba/nlg-eval).
- **BERTScore** uses the `bert-base-chinese` model, which performs best on Chinese corpora (see [bert_score](https://github.com/Tiiiger/bert_score) for details).
- **Sentence-BERT** uses `paraphrase-multilingual-MiniLM-L12-v2`.

> Please refer to the source code for exact configurations.

---

## 💡 Notes & Tips

- **Switching the LLM backbone.** If you replace the LLM backbone, you must adapt its embedding injection / replacement logic accordingly. Because ChatGLM's `generate` method does **not** support embedding inputs, we modified the corresponding `modeling_chatglm.py`. For other LLM backbones, you can perform embedding replacement directly by passing `inputs_embeds`.

---

## 📝 Citation

If this repository or dataset is helpful for your research, please cite:

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

> The official BibTeX (including full fields such as the publication address) can be copied from the [ACL Anthology page](https://aclanthology.org/2026.acl-long.2012/).

---

## 🙏 Acknowledgements

- The dataset is built upon **M3ED** (Zhao et al., 2022). All annotations are based on the publicly available M3ED dataset and were authorized by the original data owners.

---

## 📄 License

- **Code** — released under the **MIT** License; see [LICENSE](./LICENSE).
- **Dataset** — restricted to **non-commercial academic research** only; see [Getting the Dataset](#getting-the-dataset) for details.
