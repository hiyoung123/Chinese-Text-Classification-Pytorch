#!usr/bin/env python
#-*- coding:utf-8 -*-

import argparse
import os
import jieba
import pickle as pkl
from tqdm import tqdm

import pandas as pd
import numpy as np

UNK, PAD = '<UNK>', '<PAD>'
tokenizer = jieba.cut


def build_vocab(dataset, max_size, min_freq):
    vocab_dic = {}
    for line in tqdm(dataset):
        content = line[0].strip()
        if not content:
            continue

        for word in content.split():
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def load_dataset(path):
    contents = []
    word_max_len = 0
    char_max_len = 0
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            lin = lin.split('\t')
            token = tokenizer(lin[0])
            token = [x for x in token if len(x) != 0 and x != ' ']
            char_max_len = max(char_max_len, len(lin[0]))
            word_max_len = max(word_max_len, len(token))
            if len(lin) != 2:
                contents.append([' '.join(token), -1])
            else:
                contents.append([' '.join(token), int(lin[1])])
    return contents, word_max_len, char_max_len


def embedding_map(embedding):
    result = {}
    with open(embedding, encoding='utf-8') as f:
        num, embedding_dim = f.readline().split()
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            result[word] = coefs
    return result


def biGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    return (t1 * 14918087) % buckets


def triGramHash(sequence, t, buckets):
    t1 = sequence[t - 1] if t - 1 >= 0 else 0
    t2 = sequence[t - 2] if t - 2 >= 0 else 0
    return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets


def main(config):

    if not os.path.exists(config.out_dir):
        os.mkdir(config.out_dir)
        print('%s not exists, create it !' % config.out_dir)

    train, train_word_len, train_char_len = load_dataset(config.data_dir + '/train.txt')
    dev, dev_word_len, dev_char_len = load_dataset(config.data_dir + '/dev.txt')
    test, test_word_len, test_char_len = load_dataset(config.data_dir + '/test.txt')

    columns = ['text', 'label']
    train_df = pd.DataFrame(columns=columns, data=train)
    train_df.to_csv(config.out_dir + '/train.csv', index=False)

    dev_df = pd.DataFrame(columns=columns, data=dev)
    dev_df.to_csv(config.out_dir + '/dev.csv', index=False)

    test_df = pd.DataFrame(columns=columns, data=test)
    test_df.to_csv(config.out_dir + '/test.csv', index=False)

    word_max_len = max(train_word_len, dev_word_len, test_word_len)
    char_max_len = max(train_char_len, dev_char_len, test_char_len)
    print("Word max seq len is %s, char max seq len is %s" % (word_max_len , char_max_len))

    vocab = build_vocab(
        train + dev + test,
        max_size=config.max_vocab_size,
        min_freq=config.min_freq
    )
    pkl.dump(vocab, open(config.vocab_path, 'wb'))

    print("Vocab size: %s" % len(vocab))

    embedding = embedding_map(config.vector_path)
    pkl.dump(embedding, open(config.embedding_path, 'wb'))

    print('Process data done!')
    print('Save processed train data as %s' % config.out_dir + '/train.csv')
    print('Save processed dev data as %s' % config.out_dir + '/dev.csv')
    print('Save processed test data as %s' % config.out_dir + '/test.csv')
    print('Save vocab file as %s' % config.vocab_path)
    print('Save embedding file as %s' % config.embedding_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/', help="Data dir.")
    parser.add_argument('--out_dir', type=str, default='data/', help="Out dir.")
    parser.add_argument('--max_vocab_size', type=int, default=20000, help="Max vocab size.")
    parser.add_argument('--min_freq', type=int, default=0, help="Min freq.")
    parser.add_argument('--vocab_path', type=str, default='data/', help="Vocab file path.")
    parser.add_argument('--vector_path', type=str, default='data/', help="Word vector file path.")
    parser.add_argument('--embedding_path', type=str, default='data/', help="Embedding file path.")
    args = parser.parse_args()

    main(args)

