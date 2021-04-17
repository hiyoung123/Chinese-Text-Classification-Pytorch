#!usr/bin/env python
#-*- coding:utf-8 -*-

import pickle
import random
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertModel
from tensorboardX import SummaryWriter

from .models import (
    TextRNN,
    TextCNN,
    TextRCNN,
    DPCNN,
    FastText,
    BertFCModel,
    BertRNN,
    BertCNN,
    BertRCNN
)
from .datasets import EmbeddingDataset, BertDataset
from .tricks import (
    FocalLoss, DiceLoss, LabelSmoothingCrossEntropy,
    FGM,
    Lookahead,
)
from utils.log import Log


MODEL_CLASSES = {
    'TextCNN': (TextCNN, EmbeddingDataset),
    'FastText': (FastText, EmbeddingDataset),
    'TextRCNN': (TextRCNN, EmbeddingDataset),
    'TextRNN': (TextRNN, EmbeddingDataset),
    'DPCNN': (DPCNN, EmbeddingDataset),
    'BertFC': (BertFCModel, BertDataset),
    'BertCNN': (BertCNN, BertDataset),
    'BertRNN': (BertRNN, BertDataset),
    'BertRCNN': (BertRCNN, BertDataset),
}


LOSS_FUNCTIONS = {
    'focal_loss': FocalLoss(),
    'dice_loss': DiceLoss(),
    'ce_loss': CrossEntropyLoss(),
    'label_smooth': LabelSmoothingCrossEntropy()
}


class BaseTrainer:
    def __init__(self, config, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = LOSS_FUNCTIONS[config.loss_type]

        if config.adv_tpye == 'fgm':
            self.fgm = FGM(self.model)
        else:
            self.fgm = None

        if config.flooding:
            self.flooding = config.flooding
        else:
            self.flooding = None

        if config.init_weight:
            self.init_network()

        self.num_epochs = config.num_epochs
        self.start_time = 0
        self.best_score = float('-inf')
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_step = 0
        self.patience_counter = 0
        self.patience = config.patience
        self.best_model_file = config.model_path

        base_dir = config.log_dir + '/' + config.model + '_' + config.task_name + '/'
        self.train_writer = SummaryWriter(log_dir=base_dir + 'train')
        self.eval_writer = SummaryWriter(log_dir=base_dir + 'eval')
        self.logger = Log(base_dir, config.log_level)

        self.logger.info('Config', str(config))

    def train(self, train, dev):
        pass

    def train_epoch(self, data, epoch):
        pass

    def train_step(self, batch):
        batch = {key: value.to(self.device) for key, value in batch.items()}

        self.optimizer.zero_grad()
        logits, _ = self.model(batch)
        loss = self.criterion(logits, batch['label'].view(-1))

        if self.flooding:
            flooding = torch.tensor(self.flooding).to(self.device)
            loss = torch.abs(loss - flooding) + flooding

        loss.backward()

        if self.fgm:
            self.fgm.attack()
            logits_adv, _ = self.model(batch)
            loss_adv = self.criterion(logits_adv, batch['label'].view(-1))
            loss_adv.backward()
            self.fgm.restore()

        self.optimizer.step()
        _, predict = torch.max(logits, 1)
        return loss, predict

    def check_best_score(self, result):
        score = result['acc']
        loss = result['loss']
        epoch = result['epoch']
        step = result['step']
        if score <= self.best_score:
            self.patience_counter += 1
        else:
            self.best_score = score
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_step = step
            self.patience_counter = 0
            torch.save(self.model, self.best_model_file)

    def validate(self, dataset, epoch):
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
            'epoch': epoch,
            'step': 0,
        }
        return result

    def init_network(self, method='xavier', exclude='embedding'):
        for name, w in self.model.named_parameters():
            if exclude not in name:
                if 'weight' in name:
                    if method == 'xavier':
                        nn.init.xavier_normal_(w)
                    elif method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass

    def build_result(self):
        total_time = time.process_time() - self.start_time
        m, s = int(total_time/60), total_time%60
        self.logger.info('Train', 'The training result is :')
        self.logger.info('Train', 'Best Loss is %.4f' % self.best_loss)
        self.logger.info('Train', 'Best Score is %.4f' % self.best_score)
        self.logger.info('Train', 'Best Epoch is %s' % self.best_epoch)
        self.logger.info('Train', 'Best Step is %s' % self.best_step)
        self.logger.info('Train', 'The time is %s m %.2f s' % (m, s))


class EpochTrainer(BaseTrainer):

    def train(self, train, dev):
        self.start_time = time.process_time()
        for epoch in range(self.num_epochs):
            train_result = self.train_epoch(train, epoch)
            val_result = self.validate(dev, epoch)

            self.train_writer.add_scalar('Loss', train_result['loss'], epoch)
            self.train_writer.add_scalar('Acc', train_result['acc'], epoch)
            self.eval_writer.add_scalar('Loss', val_result['loss'], epoch)
            self.eval_writer.add_scalar('Acc', val_result['acc'], epoch)

            self.check_best_score(val_result)

            if self.patience_counter >= self.patience:
                break
        self.build_result()

    def train_epoch(self, train, epoch):
        self.model.train()
        total_loss = 0
        predict_list, label_list = [], []
        desc = '[Train Epoch: {0}/{1}]'.format(epoch + 1, self.num_epochs)
        batch_iterator = tqdm(train, desc=desc, ncols=100)
        for i, batch in enumerate(batch_iterator):

            loss, predict = self.train_step(batch)

            total_loss += loss.item()
            predict_list += predict.cpu().numpy().tolist()
            label_list += batch['label'].cpu().numpy().tolist()

            postfix = {'loss': total_loss / (i + 1)}
            batch_iterator.set_postfix(postfix)

        result = {
            'loss': total_loss / len(train),
            'acc': accuracy_score(label_list, predict_list),
            'epoch': epoch,
        }
        return result


class StepTrainer(BaseTrainer):
    def __init__(self, config, model):
        super(StepTrainer, self).__init__(config, model)
        self.global_step = 0
        self.log_step = config.log_step

    def train(self, train, dev):
        for epoch in range(self.num_epochs):
            if self.train_epoch((train, dev), epoch):
                break
        self.build_result()

    def train_epoch(self, data, epoch):
        self.model.train()
        train, dev = data
        total_loss, pre_loss = 0, 0
        pre_list, label_list = [], []
        desc = '[Train Epoch: {0}/{1}]'.format(epoch + 1, self.num_epochs)
        batch_iterator = tqdm(train, desc=desc, ncols=100)
        for i, batch in enumerate(batch_iterator):
            loss, predict = self.train_step(batch)

            total_loss += loss.item()
            pre_list += predict.cpu().numpy().tolist()
            label_list += batch['label'].cpu().numpy().tolist()

            postfix = {'loss': total_loss / (i + 1)}
            batch_iterator.set_postfix(postfix)

            self.global_step += 1
            if self.log_step > 0 and self.global_step % self.log_step == 0:
                val_result = self.validate(dev, epoch)
                val_result['step'] = self.global_step
                train_acc = accuracy_score(label_list, pre_list) # log_step 个 patch 的平均
                pre_list, label_list = [], []

                self.train_writer.add_scalar('Loss', (total_loss-pre_loss)/self.log_step, self.global_step)
                self.train_writer.add_scalar('Acc', train_acc, self.global_step)
                self.eval_writer.add_scalar('Loss', val_result['loss'], self.global_step)
                self.eval_writer.add_scalar('Acc', val_result['acc'], self.global_step)
                pre_loss = total_loss
                self.model.train()
                self.check_best_score(val_result)
            if self.patience_counter >= self.patience / self.log_step:
                return True
        return False


def set_seed(seed):
    # seed = 7874
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    dev = pd.read_csv(config.dev_path)

    # train, dev, _, _ = train_test_split(
    #     train, train['label'],
    #     test_size=config.split_size,
    #     stratify=train['label'],
    #     random_state=config.seed
    # )

    set_seed(config.seed)

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
    trainer.train(train, dev)