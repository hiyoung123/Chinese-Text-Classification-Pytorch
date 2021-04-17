#!usr/bin/env python
#-*- coding:utf-8 -*-

import pickle
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertModel

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


class Evaluator:
    def __init__(self, config, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model = torch.load(config.model_path, map_location=self.device)
        self.logger = Log()

    def evaluate(self, data):
        desc = '[Evaluate]'
        batch_iterator = tqdm(data, desc=desc, ncols=100)
        pre_list = []
        label_list = []
        logits_list = []
        prob_list = []
        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(batch_iterator):
                batch = {key: value.to(self.device) for key, value in batch.items()}
                logits, prob = self.model(batch)
                _, pre = torch.max(logits, 1)
                pre_list += pre.cpu().numpy().tolist()
                label_list += batch['label'].cpu().numpy().tolist()
                logits_list += logits.cpu().numpy().tolist()
                prob_list += prob.cpu().numpy().tolist()
        result = {
            'acc': accuracy_score(label_list, pre_list),
        }
        self.logger.info('Evaluate', 'evaluation score is %s' % result['acc'])
        return result['acc']


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


def run_eval(config):

    test = pd.read_csv(config.test_path)

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

    test = dataset(test, tokenizer, config.max_seq_len, True)
    test = DataLoader(test, batch_size=config.batch_size)
    evaluator = Evaluator(config, model)
    acc = evaluator.evaluate(test)
    # print(acc)
