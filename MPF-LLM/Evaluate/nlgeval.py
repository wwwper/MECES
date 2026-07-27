import json

from nlgeval import NLGEval

def chinese_char_tokenizer(text: str) -> str:
    """
    将中文字符串按字切分，并用空格连接。
    例如: "你好世界" -> "你 好 世 界"
    """
    return " ".join(list(text))

pred =[]
with open('../../Paper_experiment/results/pred/RAG_(Simple_data)MECA_chatglm_(0,8,3)_V100_Text_model_modify_output.json', 'r', encoding='utf-8') as file:
    pred_data = json.load(file)
    for i in range(0, len(pred_data)):
        for j in range(0, len(pred_data[i]["dialog"])):
            if pred_data[i]["dialog"][j]["Emotion"] != "Neutral" and pred_data[i]["dialog"][j]["Cause_utterance"]!=["无法标注"]:
                pred_data.append(pred_data[i]["dialog"][j]["Cause_summary"][0])

ref =[]
with open('./mpf_llm/dataset/MECESD_test.json', 'r', encoding='utf-8') as file:
    ref_data = json.load(file)
    for i in range(0, len(ref_data)):
        for j in range(0, len(ref_data[i]["dialog"])):
            if ref_data[i]["dialog"][j]["Emotion"] != "Neutral" and ref_data[i]["dialog"][j]["Cause_utterance"]!=["无法标注"]:
                ref_data.append(ref_data[i]["dialog"][j]["Cause_summary"][0])


tokenized_candidates = [chinese_char_tokenizer(sent) for sent in pred]
tokenized_references = [chinese_char_tokenizer(ref) for ref in ref]


nlgeval_ = NLGEval(no_glove=True, no_skipthoughts=True)
ans=nlgeval_.compute_metrics(hyp_list=tokenized_candidates ,ref_list=[tokenized_references])
print(ans)