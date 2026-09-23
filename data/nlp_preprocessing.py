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
        for i,j in enumerate(collection): # Using enumerate is much faster than collection.index()
            vocabulary[j] = i + 1
        pos = []
        neg = []
        # using zip will create intermediate tuples for every iteration , hence use normal loop
        for i in positive:
            pos.append(torch.tensor([vocabulary[k] for k in i.split()]))
        for j in negative:
            neg.append(torch.tensor([vocabulary[l] for l in j.split()]))
        return nn.utils.rnn.pad_sequence(pos+neg,padding_value=0,batch_first=True)

