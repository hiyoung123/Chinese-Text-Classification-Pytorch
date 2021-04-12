#!usr/bin/env python
#-*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset


class BertDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_seq_len, train=True):
        self.tokenizer = tokenizer
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
            text = ''.join([x for x in text.split()])
            text = self.tokenizer.tokenize(text)

            if len(text) > (self.max_seq_len - 2):
                text = text[0:(self.max_seq_len - 2)]
            text = ['[CLS]'] + text + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(text)
            token_type_ids = [0] * len(input_ids)
            padding = [0] * (self.max_seq_len - len(input_ids))
            attention_mask = [1] * len(input_ids) + padding
            token_type_ids = token_type_ids + padding
            input_ids += padding
            output = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
            }
            if self.train:
                output.update({'label': dataset['label'].iloc[i]})
            result.append({key: torch.tensor(value) for key, value in output.items()})
        return result