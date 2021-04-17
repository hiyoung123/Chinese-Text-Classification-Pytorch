#!usr/bin/env python
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class BertCNN(nn.Module):
    def __init__(self, config, bert):
        super(BertCNN, self).__init__()
        self.bert = bert
        for param in self.bert.parameters():
            param.requires_grad = True

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, self.bert.config.hidden_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_labels)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, inputs):
        outputs = self.bert(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                            token_type_ids=inputs['token_type_ids'])
        out = outputs[0]
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        prob = nn.functional.softmax(out, dim=-1)
        return out, prob