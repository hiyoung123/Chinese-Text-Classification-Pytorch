#!usr/bin/env python
#-*- coding:utf-8 -*-

import pickle
import random
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


class Trainer:
    def __init__(self, config, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = LOSS_FUNCTIONS[config.loss_type]

    def train(self, train, dev, num_epochs, patience, best_model_file):

        best_score = float('-inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            desc = '[Train Epoch: {0}/{1}]'.format(epoch + 1, num_epochs)
            train_loss, train_acc = self.train_func(train, desc)
            val_loss, val_acc = self.validate(dev)
            print('train loss = {:.4f}, train acc= {:.4f}, val loss = {:.4f}, val acc = {:.4f}'
                  .format(train_loss, train_acc, val_loss, val_acc))

            if val_acc <= best_score:
                patience_counter += 1
            else:
                best_score = val_acc
                patience_counter = 0
                # logging.info('Best score is {}'.format(best_score))
                torch.save(self.model, best_model_file)
            if patience_counter >= patience:
                # logging.info("Early stopping: patience limit reached, stopping...")
                break

    def train_func(self, dataset, desc):
        self.model.train()
        total_loss, pre_loss = 0, 0
        pre_list, label_list = [], []

        batch_iterator = tqdm(dataset, desc=desc, ncols=100)
        for i, batch in enumerate(batch_iterator):
            batch = {key: value.to(self.device) for key, value in batch.items()}

            self.optimizer.zero_grad()
            logits, _ = self.model(batch)
            loss = self.criterion(logits, batch['label'].view(-1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            _, pre = torch.max(logits, 1)

            pre_list += pre.cpu().numpy().tolist()
            label_list += batch['label'].cpu().numpy().tolist()

            postfix = {'loss': total_loss / (i + 1)}
            batch_iterator.set_postfix(postfix)

        result = {
            'loss': total_loss / len(dataset),
            'acc': accuracy_score(label_list, pre_list),
        }
        return result['loss'], result['acc']

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
        return result['loss'], result['acc']


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True


def build_embedding(config, vocab):
    embedding_matrix = np.zeros((len(vocab) + 1, config.embed_dim))
    embeddings_index = pickle.load(open(config.embedding_path, 'rb'))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return torch.Tensor(embedding_matrix)


def run_train(config):

    init_seed(config.seed)

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

    trainer = Trainer(config, model)
    trainer.train(train, dev, num_epochs=config.num_epochs, patience=config.patience, best_model_file=config.model_path)