#!usr/bin/env python
#-*- coding:utf-8 -*-

import torch.nn as nn


class TextRNN(nn.Module):
    """
        Recurrent Neural Network for Text Classification with Multi-Task Learning
    """
    def __init__(self, config):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(config.embedding, freeze=False)

        self.lstm = nn.LSTM(config.embed_dim, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_labels)

    def forward(self, inputs):
        out = self.embedding(inputs['text'])
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        prob = nn.functional.softmax(out, dim=-1)
        return out, prob