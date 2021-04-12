#!usr/bin/env python
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRCNN(nn.Module):
    """
        Recurrent Convolutional Neural Networks for Text Classification
    """
    def __init__(self, config):
        super(TextRCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(config.embedding, freeze=False)

        self.lstm = nn.LSTM(config.embed_dim, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.max_seq_len)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed_dim, config.num_labels)

    def forward(self, inputs):
        embed = self.embedding(inputs['text'])
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        prob = nn.functional.softmax(out, dim=-1)
        return out, prob