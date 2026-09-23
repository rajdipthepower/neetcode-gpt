import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        collection = set()
        vocabulary = {}
        for i,j in zip(positive,negative):
            collection.update(i.split())
            collection.update(j.split())
        collection = sorted(collection)
        for i in collection:
            vocabulary[i] = collection.index(i) + 1
        pos = []
        neg = []
        for i,j in zip(positive,negative):
            sen1 = []
            sen2 = []
            for k in i.split():
                sen1.append(vocabulary[k])
            for l in j.split():
                sen2.append(vocabulary[l])
            pos.append(torch.tensor(sen1))
            neg.append(torch.tensor(sen2))
        return nn.utils.rnn.pad_sequence(pos+neg,padding_value=0,batch_first=True)

