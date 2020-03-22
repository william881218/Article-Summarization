import torch
import json
import nltk
import re
import sys
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

word_vec_d = 50
glove_path = './glove.6B.50d.txt'

def read_glove():
    glove = open(glove_path).read().split('\n')
    glove = [line.split(' ') for line in glove]
    return {line[0]:line[1:] for line in glove}
#in order to read glove only once
glove = read_glove()
print('glove read complete')

class Article():
    def __init__(self, article, word2index, index2vec):
        self.id = article['id']
        self.sentence_cut = []
        self.tokened_text = np.ndarray(shape=(1,0))
        self.text_size = 0
        for line in article['text']:
            self.sentence_cut.append((self.text_size, self.text_size + len(line)))
            self.text_size += len(line)
            for word in line:
                self.tokened_text = np.append(self.tokened_text, [word2index[word]])
                #self.tokened_text = np.append(self.tokened_text, [index2vec[word2index[word]]])
        self.extractive_gt = article['extractive_summary']
        self.abstractive_gt = [word2index[word] for word in article['summary']]
        #self.abstractive_gt = [index2vec[word2index[word]] for word in article['summary']]
    def show(self):
        print('id: {}, text_size: {}'.format(self.id, self.text_size))
        print('tokened_text:')
        for word in self.tokened_text:
            print('{}, '.format(word), end='')
        print('')
        for i, cut in enumerate(self.sentence_cut):
            print('Sentence {}: {}'.format(i, self.tokened_text[cut[0]: cut[1]]))
        print('extractive_gt: {}'.format(self.extractive_gt))
        print('abstractive_gt: {}'.format(self.abstractive_gt))

"""
Json file keys:
id, text, summary, sent_bounds, extractive_summary
"""



class ArticleDataset(Dataset):
    def __init__(self, dataset_path):
        lines = open(dataset_path).read().lower().strip().split('\n')
        datas = [json.loads(line) for line in lines]
        self.articles = []
        self.word2index = {}
        self.index2vec = [[0 for i in range(word_vec_d)]]
        for data in datas:
            sys.stderr.write(str(data['id'])+'\n')
            data['text'] = [nltk.word_tokenize(data['text'][begin:end]) for begin, end in data['sent_bounds'] ]
            if len(data['text'][0]) == 0:
                continue
            data['summary'] = [word for word in nltk.word_tokenize(data['summary'])]
            for line in data['text']:
                for word in line:
                    if word not in self.word2index:
                        self.word2index[word] = len(self.index2vec)
                        try:
                            self.index2vec += [glove[word]]
                        except KeyError:
                            self.index2vec += [np.random.standard_normal(word_vec_d).tolist()]
            for word in data['summary']:
                if word not in self.word2index:
                    self.word2index[word] = len(self.index2vec)
                    try:
                        self.index2vec += [glove[word]]
                    except KeyError:
                        self.index2vec += [np.random.standard_normal(word_vec_d).tolist()]
            self.articles.append(Article(data, self.word2index, self.index2vec))
        self.vocab_size = len(self.index2vec)
        self.index2vec = np.array(self.index2vec)
    def __len__(self):
        return len(self.articles)
    def __getitem__(self, idx):
        article = self.articles[int(idx)]
        #extractive_gt = np.zeros(shape=(article.text_size, word_vec_d))
        extractive_gt = np.zeros(shape=(article.text_size))
        ext_cut = article.sentence_cut[int(article.extractive_gt)]
        extractive_gt[ext_cut[0]:ext_cut[1]] = 1.
        return (torch.from_numpy(article.tokened_text.reshape([article.text_size]).astype(dtype='int32')),
                torch.from_numpy(extractive_gt))
    def word_embedding(self):
        return self.index2vec
