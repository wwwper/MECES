import torch
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from typing import Dict, List

import torch.nn as nn
import torch.nn.functional as F


VIDEO_PLACEHOLDER = "<video_placeholder>"
AUDIO_PLACEHOLDER = "<audio_placeholder>"


image_features_dir=r"multimodal_features/video_features.pt"
audio_features_dir=r"multimodal_features/audio_features.pt"

image_features = torch.load(image_features_dir)
audio_features = torch.load(audio_features_dir)


def sanity_check(tokens: List[int], target: List[int], tokenizer: PreTrainedTokenizer):
    print("Sanity Check >>>>>>>>>>>>>")
    for t, m in zip(tokens, target):
        decoded =  tokenizer.tokenizer.index_special_tokens[t] \
            if t in tokenizer.tokenizer.index_special_tokens \
            else tokenizer.decode([t])
        # print("%20s: %6d -> %6d" % (repr(decoded), t, m))
    print("<<<<<<<<<<<<< Sanity Check")

    assert len(tokens) == len(target), f"length mismatch: {len(tokens)} vs {len(target)}"


class InputOutputDataset(Dataset):
    def __init__(self, data: List[dict], tokenizer: PreTrainedTokenizer, max_source_length: int, max_target_length: int):
        super(InputOutputDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = max_source_length + max_target_length + 1
        self.data = data


    def feature_read(self, image_features_dic, audio_features_dic, key_id_list):
        multimodal_tensor_list = []

        for key in key_id_list:
            image_tensor = image_features_dic[key]
            image_tensor=image_tensor.squeeze(0)

            audio_tensor = audio_features_dic[key]
            audio_tensor=audio_tensor.squeeze(0)

            multimodal_tensor_list.append(image_tensor)
            multimodal_tensor_list.append(audio_tensor)
        return multimodal_tensor_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> dict:
        # 获取自定义token的ID
        video_token_id = self.tokenizer.convert_tokens_to_ids(VIDEO_PLACEHOLDER)
        audio_token_id = self.tokenizer.convert_tokens_to_ids(AUDIO_PLACEHOLDER)

        ##dataset返回值内容
        data_item = self.data[i]

        a_ids = self.tokenizer.encode(text=data_item['context'], add_special_tokens=True, truncation=True,
                                         max_length=self.max_source_length)
        b_ids = self.tokenizer.encode(text=data_item['target'], add_special_tokens=False, truncation=True,
                                    max_length=self.max_target_length)

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

        pad_len = self.max_seq_length - len(input_ids)

        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [self.tokenizer.pad_token_id] * pad_len

        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        # 多模态token处理

        indices = [idx for idx, token_id in enumerate(input_ids) if token_id in [video_token_id, audio_token_id]]# 定位多模态占位符索引
        if len(indices)==0:
            raise AssertionError(f"multimodal token none!")

        features_key_list = data_item["multimodal_features_key_list"]  

        features_key_list_length=len(data_item["multimodal_features_key_list"]) ##一个需要嵌入的utt的数量，每个utt对应两个token
        modality_list=['video' if i % 2 == 0 else 'audio' for i in range(features_key_list_length*2)]

        multimodal_feats_list = self.feature_read(image_features, audio_features, features_key_list) # 这是一个 tensor 列表

        assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"
        output_dic={
            "input_ids": input_ids,
            "labels": labels,
            "raw_multimodal_features": multimodal_feats_list,
            "modality_types": modality_list, # 模态类型列表
            "multimodal_indices":torch.tensor(indices, dtype=torch.long),
        }

        return output_dic
