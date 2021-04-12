#!usr/bin/env python
#-*- coding:utf-8 -*-

import torch.nn as nn


class BertClassificationModel(nn.Module):
    def __init__(self, config, bert):
        super(BertClassificationModel, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_labels)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, inputs):
        outputs = self.bert(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                            token_type_ids=inputs['token_type_ids'])
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        prob = nn.functional.softmax(logits, dim=-1)
        return logits, prob