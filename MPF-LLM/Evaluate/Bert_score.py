import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import json

from bert_score import score

bert_path="Huggingface_model/bert-base-chinese"
def calculate_bertscore(candidate, reference, model_type=bert_path, lang='zh'):
    """计算 BERTScore F1 分数"""
    # 转换为列表形式，以适应库的输入要求
    candidates = candidate
    references = reference

    # score 函数返回 P (Precision), R (Recall), F1 (F1-score)
    # 首次运行时会自动下载模型，可能需要一些时间
    P, R, F1 = score(candidates, references, num_layers=8, lang=lang, model_type=model_type, verbose=True)

    # return F1
    # 返回平均 F1 分数
    return F1.mean().item()



pred =[]
# 请确保这里的路径是正确的
with open('../results/pred/MECES_pred.json', 'r', encoding='utf-8') as file:
    pred_data = json.load(file)
    for i in range(0, len(pred_data)):
        for j in range(0, len(pred_data[i]["dialog"])):
            if pred_data[i]["dialog"][j]["Emotion"] != "Neutral" and pred_data[i]["dialog"][j]["Cause_utterance"]!=["无法标注"]:
                pred_data.append(pred_data[i]["dialog"][j]["Cause_summary"][0])

ref =[]


with open('../mpf_llm/dataset/MECESD_test.json', 'r', encoding='utf-8') as file:
    ref_data = json.load(file)
    for i in range(0, len(ref_data)):
        for j in range(0, len(ref_data[i]["dialog"])):
            if ref_data[i]["dialog"][j]["Emotion"] != "Neutral" and ref_data[i]["dialog"][j]["Cause_utterance"]!=["无法标注"]:
                ref_data.append(ref_data[i]["dialog"][j]["Cause_summary"][0])


bertscore_f1_zh = calculate_bertscore(
    pred,
    ref,
    model_type=bert_path,  # 建议使用中文模型
    lang='zh'
)
print(f"BERTScore (Chinese) F1 Score: {bertscore_f1_zh:.4f}")

