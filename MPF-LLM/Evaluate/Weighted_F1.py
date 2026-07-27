import json
from collections import defaultdict

def calculate_ece_f1(predictions, ground_truths):
    """
    计算全局 Micro-F1 (所有样本混合计算)
    """
    tp, fp, fn = 0, 0, 0

    for pred, true in zip(predictions, ground_truths):
        pred_set = set(pred)
        true_set = set(true)

        # 计算交集（TP）
        correct_hits = pred_set & true_set
        tp += len(correct_hits)

        # 计算FP (预测有但真实没有)
        fp += len(pred_set - true_set)

        # 计算FN (真实有但预测没有)
        fn += len(true_set - pred_set)

    # 避免除以零
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


# ### 新增函数：计算 Weighted F1 ###
def calculate_weighted_ece_f1(predictions, ground_truths, emotion_labels):
    """
    计算按情绪类别加权的 Weighted F1
    """
    # 按情绪类别存储统计信息: {'happy': {'tp': 0, 'pred_c': 0, 'true_c': 0}, ...}
    class_stats = defaultdict(lambda: {'tp': 0, 'pred_c': 0, 'true_c': 0})

    for pred, true, emo in zip(predictions, ground_truths, emotion_labels):
        pred_set = set(pred)
        true_set = set(true)

        # 累计该情绪类别的 TP, 预测总数, 真实总数
        class_stats[emo]['tp'] += len(pred_set & true_set)
        class_stats[emo]['pred_c'] += len(pred_set)
        class_stats[emo]['true_c'] += len(true_set)

    total_weighted_f1 = 0
    total_true_support = 0  # 所有真实原因句子的总数

    # 打印每个情绪的详细信息（可选）
    print("\n--- 各情绪类别详细指标 ---")

    for emo, stats in class_stats.items():
        tp = stats['tp']
        pred_c = stats['pred_c']
        true_c = stats['true_c']

        p = tp / pred_c if pred_c > 0 else 0.0
        r = tp / true_c if true_c > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        # Weighted F1 核心：F1 * 该类别的真实数量权重
        total_weighted_f1 += f1 * true_c
        total_true_support += true_c

        print(f"Emotion: {emo:<12} | F1: {f1:.4f} | Support: {true_c}")

    # 计算加权平均
    weighted_f1 = total_weighted_f1 / total_true_support if total_true_support > 0 else 0.0

    return weighted_f1


# ================= 数据读取部分 =================

pred = []
# 请确保这里的路径是正确的
with open('../results/pred/MECES_pred.json','r', encoding='utf-8') as file:
    pred_data = json.load(file)
    for i in range(0, len(pred_data)):
        for j in range(0, len(pred_data[i]["dialog"])):
            if pred_data[i]["dialog"][j]["Emotion"] != "Neutral" and pred_data[i]["dialog"][j]["Cause_utterance"] != ["无法标注"]:
                pred.append(pred_data[i]["dialog"][j]["Cause_utterance"])

label = []
emotions_list = []  # ### 新增：用于存储对应的真实情绪标签 ###

with open('../mpf_llm/dataset/MECESD_test.json', 'r', encoding='utf-8') as file:
    label_data = json.load(file)
    for i in range(0, len(label_data)):
        for j in range(0, len(label_data[i]["dialog"])):
            if label_data[i]["dialog"][j]["Emotion"] != "Neutral" and label_data[i]["dialog"][j]["Cause_utterance"] != ["无法标注"]:
                label.append(label_data[i]["dialog"][j]["Cause_utterance"])
                # 注意：这里读取的是Dataset(Ground Truth)里的情绪，这是计算权重的标准
                emotions_list.append(label_data[i]["dialog"][j]["Emotion"])

# ================= 计算与输出 =================


p, r, f1 = calculate_ece_f1(pred, label)
print(f"Global Precision: {p:.4f}, Recall: {r:.4f}, Micro F1: {f1:.4f}")

assert len(pred) == len(label) == len(emotions_list), "错误：预测值、真实值或情绪标签的数量不一致！"

w_f1 = calculate_weighted_ece_f1(pred, label, emotions_list)
print(f"\nWeighted F1: {w_f1:.4f}")
