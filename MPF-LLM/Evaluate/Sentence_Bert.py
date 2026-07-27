import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


print("正在加载模型...")
model = SentenceTransformer('/data/wjk/Huggingface_model/paraphrase-multilingual-MiniLM-L12-v2')
print("模型加载完成。")

pred =[]
# 请确保这里的路径是正确的
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


# 检查列表长度是否一致
assert len(pred) == len(ref), "生成句子和参考句子的数量必须相同"

# 3. 对两组句子分别进行编码
print("\n正在对句子进行编码...")
embeddings_gen = model.encode(pred, convert_to_tensor=True)
embeddings_ref = model.encode(ref, convert_to_tensor=True)
print("编码完成。")

# 4. 【核心区别】使用 util.pairwise_cos_sim 进行一一对应的相似度计算
#    这个函数只会计算 embeddings_gen[i] 和 embeddings_ref[i] 之间的分数。
pairwise_scores = util.pairwise_cos_sim(embeddings_gen, embeddings_ref)

# 5. 打印结果
print("\n一一对应的语义相似度得分:")
for i in tqdm(range(len(pred))):
    score = pairwise_scores[i].item() # 直接从一维张量中取值

average_score = pairwise_scores.mean().item()

print(f"\n==============================")
print(f"所有句子对的平均相似度得分: {average_score:.4f}")
print(f"==============================")