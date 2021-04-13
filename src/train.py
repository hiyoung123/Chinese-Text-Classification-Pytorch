#!usr/bin/env python
#-*- coding:utf-8 -*-

import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
from torch.optim import AdamW, Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertModel


from .models import (
    TextRNN,
    TextCNN,
    TextRCNN,
    BertClassificationModel
)
from .datasets import EmbeddingDataset, BertDataset


MODEL_CLASSES = {
    'TextCNN': (TextCNN, EmbeddingDataset),
    # 'FastText': (FastText, EmbeddingDataset),
    'TextRCNN': (TextRCNN, EmbeddingDataset),
    'TextRNN': (TextRNN, EmbeddingDataset),
    'BertFC': (BertClassificationModel, BertDataset),
}


LOSS_FUNCTIONS = {
    # 'focal_loss': FocalLoss(),
    # 'dice_loss': DiceLoss(),
    'ce_loss': CrossEntropyLoss(),
    # 'label_smooth': LabelSmoothingCrossEntropy()
}


class BaseTrainer:
    def __init__(self, config, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = LOSS_FUNCTIONS[config.loss_type]

        self.best_score = float('-inf')
        self.patience_counter = 0
        self.patience = config.patience
        self.best_model_file = config.model_path

    def train(self, train, dev, num_epochs):
        pass

    def train_epoch(self, data, desc):
        pass

    def train_step(self, batch):
        self.optimizer.zero_grad()
        logits, _ = self.model(batch)
        loss = self.criterion(logits, batch['label'].view(-1))
        loss.backward()
        self.optimizer.step()
        _, predict = torch.max(logits, 1)
        return loss, predict

    def check_best_score(self, result):
        score = result['acc']
        if score <= self.best_score:
            self.patience_counter += 1
        else:
            self.best_score = score
            self.patience_counter = 0
            torch.save(self.model, self.best_model_file)

    def validate(self, dataset):
        self.model.eval()
        total_loss = 0
        pre_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(dataset):
                batch = {key: value.to(self.device) for key, value in batch.items()}

                logits, _ = self.model(batch)
                loss = self.criterion(logits, batch['label'].view(-1))
                total_loss += loss.item()
                _, pre = torch.max(logits, 1)
                pre_list += pre.cpu().numpy().tolist()
                label_list += batch['label'].cpu().numpy().tolist()

        result = {
            'loss': total_loss / len(dataset),
            'acc': accuracy_score(label_list, pre_list),
        }
        return result


class EpochTrainer(BaseTrainer):

    def train(self, train, dev, num_epochs):
        for epoch in range(num_epochs):
            desc = '[Train Epoch: {0}/{1}]'.format(epoch + 1, num_epochs)
            train_result = self.train_epoch(train, desc)
            val_result = self.validate(dev)

            self.check_best_score(val_result)

            if self.patience_counter >= self.patience:
                break

    def train_epoch(self, train, desc):
        self.model.train()
        total_loss = 0
        predict_list, label_list = [], []

        batch_iterator = tqdm(train, desc=desc, ncols=100)
        for i, batch in enumerate(batch_iterator):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            loss, predict = self.train_step(batch)

            total_loss += loss.item()
            predict_list += predict.cpu().numpy().tolist()
            label_list += batch['label'].cpu().numpy().tolist()

            postfix = {'loss': total_loss / (i + 1)}
            batch_iterator.set_postfix(postfix)

        result = {
            'loss': total_loss / len(train),
            'acc': accuracy_score(label_list, predict_list),
        }
        return result['loss'], result['acc']


class StepTrainer(BaseTrainer):
    def __init__(self, config, model):
        super(StepTrainer, self).__init__(config, model)
        self.global_step = 0
        self.log_step = config.log_step

    def train(self, train, dev, num_epochs):
        for epoch in range(num_epochs):
            desc = '[Train Epoch: {0}/{1}]'.format(epoch + 1, num_epochs)
            if self.train_epoch((train, dev), desc):
                break

    def train_epoch(self, data, desc):
        self.model.train()
        train, dev = data
        total_loss = 0
        pre_list, label_list = [], []

        batch_iterator = tqdm(train, desc=desc, ncols=100)
        for i, batch in enumerate(batch_iterator):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            loss, predict = self.train_step(batch)

            total_loss += loss.item()
            pre_list += predict.cpu().numpy().tolist()
            label_list += batch['label'].cpu().numpy().tolist()

            postfix = {'loss': total_loss / (i + 1)}
            batch_iterator.set_postfix(postfix)

            self.global_step += 1
            if (self.log_step > 0 and self.global_step % self.log_step == 0) or self.global_step == len(batch_iterator):
                dev_result = self.validate(dev)
                train_acc = accuracy_score(label_list, pre_list) # log_step 个 patch 的平均
                pre_list, label_list = [], []

                self.model.train()
                self.check_best_score(dev_result)
            if self.patience_counter >= self.patience / self.log_step:
                return True
        return False


def build_embedding(config, vocab):
    embedding_matrix = np.zeros((len(vocab) + 1, config.embed_dim))
    embeddings_index = pickle.load(open(config.embedding_path, 'rb'))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return torch.Tensor(embedding_matrix)


def run_train(config):

    train = pd.read_csv(config.train_path)
    train, dev, _, _ = train_test_split(
        train, train['label'],
        test_size=config.split_size,
        stratify=train['label'],
        random_state=config.seed
    )

    model, dataset = MODEL_CLASSES[config.model]
    if config.get('embedding_path', False):
        tokenizer = pickle.load(open(config.vocab_path, 'rb'))
        config['embedding'] = build_embedding(config, tokenizer)
        model = model(config)
    else:
        tokenizer = BertTokenizer.from_pretrained(config.pre_trained_model + '/vocab.txt')
        bert_config = BertConfig.from_pretrained(config.pre_trained_model + '/bert_config.json')
        bert = BertModel.from_pretrained(config.pre_trained_model + '/pytorch_model.bin', config=bert_config)
        model = model(bert=bert, config=config)

    train = dataset(train, tokenizer, config.max_seq_len, True)
    train = DataLoader(train, batch_size=config.batch_size)
    dev = dataset(dev, tokenizer, config.max_seq_len, True)
    dev = DataLoader(dev, batch_size=config.batch_size)

    if config.save_by_step:
        trainer = StepTrainer(config, model)
    else:
        trainer = EpochTrainer(config, model)
    trainer.train(train, dev, num_epochs=config.num_epochs)