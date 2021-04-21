#!usr/bin/env python
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastText(nn.Module):
    """
        Bag of Tricks for Efficient Text Classification
    """
    def __init__(self, config):
        super(FastText, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(config.embedding, freeze=False)
        self.embedding_ngram2 = nn.Embedding(config.ngram_size, config.embed_dim)
        self.embedding_ngram3 = nn.Embedding(config.ngram_size, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed_dim * 3, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs):

        out_word = self.embedding(inputs['text'])
        out_bigram = self.embedding_ngram2(inputs['bigram'])
        out_trigram = self.embedding_ngram3(inputs['trigram'])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        prob = nn.functional.softmax(out, dim=-1)
        return out, prob