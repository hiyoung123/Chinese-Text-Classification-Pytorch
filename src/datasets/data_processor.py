#!usr/bin/env python
#-*- coding:utf-8 -*-

from torch.utils.data import DataLoader

from .bert_dataset import BertDataset
from .embedding_dataset import EmbeddingDataset


DATASETS = {
    'bert': BertDataset,
    'embedding': EmbeddingDataset
}


class DataProcessor:
    def __init__(self, tokenizer, max_seq_len, batch_size, data_type):
        self.dataset = DATASETS[data_type]
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def build_batch(self, data, do_train=True):
        data = self.dataset(data, self.tokenizer, self.max_seq_len, do_train)
        return DataLoader(data, batch_size=self.batch_size)