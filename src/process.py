#!usr/bin/env python
#-*- coding:utf-8 -*-

import argparse
import os
import jieba
import pickle as pkl
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

UNK, PAD = '<UNK>', '<PAD>'
tokenizer = jieba.cut


def build_vocab(dataset, max_size, min_freq):
    vocab_dic = {}
    for i, content in enumerate(dataset['text']):
        content = content.strip()
        if not content:
            continue

        for word in content.split():
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx+2 for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: 1, PAD: 0})
    return vocab_dic


def load_dataset(path):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            lin = lin.split('\t')
            token = tokenizer(lin[0])
            token = [x for x in token if len(x) != 0 and x != ' ']
            if len(lin) != 2:
                contents.append([' '.join(token), -1000])
            else:
                contents.append([' '.join(token), int(lin[1])])
    return contents


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


def show_plt(data, config):
    print('Word len describe:')
    data['text_len'] = data['text'].apply(lambda x: len(x.split()))
    print(data['text_len'].describe())
    plt.hist(data['text_len'], bins=200)
    plt.xlabel('sentence-length')
    plt.ylabel('category-number')
    plt.title('Word len describe')
    plt.savefig(config.out_dir + '/word.jpg')
    plt.show()

    print('Char len describe:')
    data['text_len'] = data['text'].apply(lambda x: len(''.join(x.split())))
    print(data['text_len'].describe())
    plt.hist(data['text_len'], bins=200)
    plt.xlabel('sentence-length')
    plt.ylabel('category-number')
    plt.title('Char len describe')
    plt.savefig(config.out_dir + '/char.jpg')
    plt.show()

    print('Label describe')
    data['label'].value_counts().plot(kind='bar')
    plt.title('News class count')
    plt.xlabel("category")
    plt.title('Label describe, Note: -1000 is test dataset label, it is invalid !')
    plt.text(0.5, 0, 'category')
    plt.savefig(config.out_dir + '/label.jpg')
    plt.show()


def main(config):

    if not os.path.exists(config.out_dir):
        os.mkdir(config.out_dir)
        print('%s not exists, create it !' % config.out_dir)

    train = load_dataset(config.data_dir + '/train.txt')
    dev = load_dataset(config.data_dir + '/dev.txt')
    test = load_dataset(config.data_dir + '/test.txt')

    columns = ['text', 'label']
    train_df = pd.DataFrame(columns=columns, data=train)
    train_df.to_csv(config.out_dir + '/train.csv', index=False)

    dev_df = pd.DataFrame(columns=columns, data=dev)
    dev_df.to_csv(config.out_dir + '/dev.csv', index=False)

    test_df = pd.DataFrame(columns=columns, data=test)
    test_df.to_csv(config.out_dir + '/test.csv', index=False)

    data = train_df.append(dev_df)
    data = data.append(test_df)

    show_plt(data, config)

    vocab = build_vocab(
        dataset=data,
        max_size=config.max_vocab_size,
        min_freq=config.min_freq
    )
    pkl.dump(vocab, open(config.vocab_path, 'wb'))

    print("Vocab size: %s" % len(vocab))

    print('Process embedding map.')
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

