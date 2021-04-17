#!usr/bin/env python
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class BertRCNN(nn.Module):
    def __init__(self, config, bert):
        super(BertRCNN, self).__init__()
        self.bert = bert
        for param in self.bert.parameters():
            param.requires_grad = True

        self.lstm = nn.LSTM(self.bert.config.hidden_size, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.max_seq_len)
        self.fc = nn.Linear(config.hidden_size * 2 + self.bert.config.hidden_size, config.num_labels)

    def forward(self, inputs):
        outputs = self.bert(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                            token_type_ids=inputs['token_type_ids'])
        out = outputs[0]
        out, _ = self.lstm(out)
        out = torch.cat((outputs[0], out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        prob = nn.functional.softmax(out, dim=-1)
        return out, prob