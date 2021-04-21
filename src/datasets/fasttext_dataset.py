#!usr/bin/env python
#-*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset

ngram_vocab_size = 250499


class FastTextDataset(Dataset):
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
            bigram = [biGramHash(text, i, ngram_vocab_size) for i, _ in enumerate(text)]
            trigram = [triGramHash(text, i, ngram_vocab_size) for i, _ in enumerate(text)]

            padding = [0] * (self.max_seq_len - len(text))

            text += padding
            bigram += padding
            trigram += padding

            output = {
                'text': text,
                'bigram': bigram,
                'trigram': trigram,
            }
            if self.train:
                output.update({'label': dataset['label'].iloc[i]})
            result.append({key: torch.tensor(value) for key, value in output.items()})
        return result



def biGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    return (t1 * 14918087) % buckets


def triGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    t2 = sequence[t - 2] if t - 2 >= 0 else 0
    return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets