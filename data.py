import torch
import json
import nltk
import re
import sys
import numpy as np

word_vec_d = 50
glove_path = './glove.6B.50d.txt'

def read_glove():
    glove = open(glove_path).read().split('\n')
    glove = [line.split(' ') for line in glove]
    return {line[0]:np.array(line[1:]) for line in glove}
glove = read_glove()

class Article():
    def __init__(self, article, word2index, index2vec):
        self.id = article['id']
        self.sentence_cut = []
        self.tokened_text = []
        self.text_size = 0
        for line in article['text']:
            self.sentence_cut.append((self.text_size, self.text_size + len(line)))
            self.text_size += len(line)
            for word in line:
                #self.tokened_text.append(word2index[word])
                self.tokened_text.append(index2vec[word2index[word]])
        self.extractive_gt = article['extractive_summary']
        #self.abstractive_gt = [word2index[word] for word in article['summary']]
        self.abstractive_gt = [index2vec[word2index[word]] for word in article['summary']]
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



class ArticleDataset():
    def __init__(self, dataset_path):
        lines = open(dataset_path).read().lower().strip().split('\n')
        datas = [json.loads(line) for line in lines]
        self.articles = []
        self.word2index = {}
        self.index2vec = []
        for data in datas:
            sys.stderr.write(str(data['id'])+'\n')
            data['text'] = [nltk.word_tokenize(data['text'][begin:end]) for begin, end in data['sent_bounds'] ]
            data['summary'] = [word for word in nltk.word_tokenize(data['summary'])]
            for line in data['text']:
                for word in line:
                    if word not in self.word2index:
                        self.word2index[word] = len(self.index2vec)
                        try:
                            self.index2vec.append(glove[word])
                        except KeyError:
                            self.index2vec.append(np.random.normal(size=word_vec_d, scale=0.2))
            for word in data['summary']:
                if word not in self.word2index:
                    self.word2index[word] = len(self.index2vec)
                    try:
                        self.index2vec.append(glove[word])
                    except KeyError:
                        self.index2vec.append(np.random.normal(size=word_vec_d, scale=0.2))

            self.articles.append(Article(data, self.word2index, self.index2vec))
        pass
    def __len__(self):
        return len(self.articles)
    def __getitem__(self, idx):
        return self.articles[int(idx)]
