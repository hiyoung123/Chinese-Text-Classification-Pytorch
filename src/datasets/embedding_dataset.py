#!usr/bin/env python
#-*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, dataset, vocab, max_seq_len, train=True):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.train = train
        self.dataset = self.process_data(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {key: value for key, value in self.dataset[idx].items()}

    def process_data(self, dataset):
        result = []
        for i, text in enumerate(dataset['text']):
            ''' 分词 ？'''
            text = [x for x in text.split()]

            if len(text) > self.max_seq_len:
                text = text[0:self.max_seq_len]

            text = [self.vocab[x] for x in text if x in self.vocab]
            padding = [0] * (self.max_seq_len - len(text))
            text += padding

            output = {
                'text': text,
            }
            if self.train:
                output.update({'label': dataset['label'].iloc[i]})
            result.append({key: torch.tensor(value) for key, value in output.items()})
        return result