"""
构建MECES的微调数据集。
"""

import json
from typing import Any, Dict, List, Optional, Tuple


# ==================== 配置 ====================

# 上下文窗口大小（以目标话语为中心，左右各取多少条）
LEFT_WINDOW = 8
RIGHT_WINDOW = 3

# 多模态占位符
MULTIMODAL_PLACEHOLDER = "<multimodal><multimodal_placeholder></multimodal>"

# 提示词前缀。

PROMPT_PREFIX = "你是情感分析和情绪原因识别方面的专家。我将给你提供一段包含两名说话者的“对话上下文”及其相应的情绪标签、多模态信息。你的任务是从中识别出导致“目标情绪话语”产生特定情绪的“原因话语”，并输出这些原因话语的索引列表，然后生成一段简洁的情绪-原因总结。 "

# 输入 / 输出路径
TRAIN_INPUT = "../dataset/MECESD_train.json"
VAL_INPUT = "../dataset/MECESD_val.json"
TEST_INPUT = "../dataset/MECESD_test.json"
_OUTPUT_DIR = "../dataset/finetune_prompt_data"
TRAIN_OUTPUT = f"{_OUTPUT_DIR}/MECES_multimodal_train_finetune.json"
VAL_OUTPUT = f"{_OUTPUT_DIR}/MECES_multimodal_val_finetune.json"
TEST_OUTPUT = f"{_OUTPUT_DIR}/MECES_multimodal_test_finetune.json"

NEUTRAL_EMOTION = "Neutral"
UNLABELED_CAUSE = ["无法标注"]


def get_context_window(index: int, total: int, left_window: int, right_window: int) -> Tuple[int, int]:
    """返回以 index 为中心、裁剪到 [0, total-1] 范围内的窗口左右边界。"""
    left = max(0, index - left_window)
    right = min(total - 1, index + right_window)
    return left, right


def merge_utterances(utterances: List[str], left: int, right: int) -> str:
    """把 utterances[left:right+1] 用换行拼接成一段上下文文本。"""
    left = max(0, left)
    right = min(len(utterances) - 1, right)
    return "\n".join(utterances[left:right + 1])


def build_utterance_strings(dialog: List[Dict[str, Any]]) -> List[str]:
    """把每条对话渲染成形如
    'U3_surprise. Speaker:...<multimodal><multimodal_placeholder></multimodal> ' 的字符串。
    """
    utterances = []
    for i, utt in enumerate(dialog):
        text = utt["Text"]
        speaker = utt["Speaker"]
        emotion = utt["Emotion"]
        utterances.append(
            f"U{i + 1}_{emotion}. {speaker}:{text}{MULTIMODAL_PLACEHOLDER} "
        )
    return utterances


def build_feature_keys(conversation_id: Any, left: int, right: int, target_number: int) -> List[str]:
    """构建候选窗口内各话语的多模态特征 key，并在末尾追加目标话语的 key。"""
    keys = [f"{conversation_id}_{utt_id}" for utt_id in range(left + 1, right + 2)]
    keys.append(f"{conversation_id}_{target_number}")
    return keys


def find_utterance_by_uid(dialog: List[Dict[str, Any]], uid: int) -> Optional[Dict[str, Any]]:
    """按 uid 在 dialog 中查找对应话语，找不到返回 None。"""
    for entry in dialog:
        if entry["Uid"] == uid:
            return entry
    return None


def build_target_strings(entry: Dict[str, Any], only_first_summary: bool = False) -> List[str]:
    """根据目标话语的原因标注，生成一个或多个 target 字符串。

    格式为 '[U1,U3],<原因总结>'。正常情况下 Cause_summary 为 1 或 2 条。
    当 only_first_summary=True 时，只取第一条，输出恒为 1 条。
    """
    cause_list = entry["Cause_utterance"]
    cause_prefix = "[" + ",".join(f"U{n}" for n in cause_list) + "]" + ","

    summaries = entry["Cause_summary"]
    if len(summaries) not in (1, 2):
        raise AssertionError("Cause_summary 长度必须为 1 或 2")

    if only_first_summary:
        summaries = summaries[:1]

    return [cause_prefix + summary for summary in summaries]


# ==================== 主逻辑 ====================
def build_dataset(data: List[Dict[str, Any]], prefix: str, output_path: str, only_first_summary: bool) -> List[Dict[str, Any]]:

    conversations: List[Dict[str, Any]] = []

    for lines in data:
        conversation_id = lines["id"]
        dialog = lines["dialog"]
        utterance_strings = build_utterance_strings(dialog)
        total = len(utterance_strings)

        for i, utt_str in enumerate(utterance_strings):
            entry = dialog[i]
            emotion = entry["Emotion"]

            # 跳过中性情绪，以及原因无法标注的话语
            if emotion == NEUTRAL_EMOTION or entry["Cause_utterance"] == UNLABELED_CAUSE:
                continue

            utt_number = i + 1
            left, right = get_context_window(i, total, LEFT_WINDOW, RIGHT_WINDOW)
            candidate_utterance = merge_utterances(utterance_strings, left, right)
            feature_keys = build_feature_keys(conversation_id, left, right, utt_number)
            context = f"{prefix}### 对话上下文:{candidate_utterance} ### 目标情绪话语: {utt_str} ### 输出: "

            target_entry = find_utterance_by_uid(dialog, utt_number)
            if target_entry is None:
                continue
            for target in build_target_strings(target_entry, only_first_summary):
                conversations.append({
                    "context": context,
                    "target": target,
                    "multimodal_features_key_list": feature_keys,
                })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=6, ensure_ascii=False)

    print("Successfully build data!")
    return conversations


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    # 训练集
    train_data = load_json(TRAIN_INPUT)
    build_dataset(train_data, PROMPT_PREFIX, TRAIN_OUTPUT, only_first_summary=False)

    # 验证集
    val_data = load_json(VAL_INPUT)
    build_dataset(val_data, PROMPT_PREFIX, VAL_OUTPUT, only_first_summary=True)

    # 测试集
    test_data = load_json(TEST_INPUT)
    build_dataset(test_data, PROMPT_PREFIX, TEST_OUTPUT, only_first_summary=True)


if __name__ == "__main__":
    main()