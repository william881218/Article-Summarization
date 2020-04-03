import torch
import json
import nltk
import re
import sys
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
word_vec_d = 50
SOS_ID = 0
EOS_ID = 1
PAD_ID = 2
glove_path = '../glove.6B.50d.txt'
softmax = nn.Softmax(dim=1)
prediction_softmax = nn.Softmax(dim=0)

#Read in word embedding
def read_glove():
    glove = open(glove_path).read().split('\n')
    glove = [line.split(' ') for line in glove]
    return {line[0]:[float(x) for x in line[1:]] for line in glove}

#read glove only once
print('start glove reading...')
glove = read_glove()
print('complete glove reading.')

class Article():
    def __init__(self, article, word2index, index2vec, idx, predicting=False):
        self.id = article['id']
        self.index = int(idx)
        self.text_size = len(article['text'])
        a = np.array([index2vec[word2index[word]] for word in article['text']]).astype('float32')
        self.text = torch.from_numpy(a)
        self.summary_size = 0 if predicting else len(article['summary']) + 1
        self.summary = None if predicting else torch.Tensor([word2index[word] for word in article['summary']] + [EOS_ID])

    def show(self):
        print('id: {}, index: {}, text_size: {}, summary_size: {}'.format(self.id, self.index, self.text_size, self.summary_size))
        print('text:')
        print(self.text)
        print('')
        print('summary: ')
        print(self.summary)


"""
Json file keys:
id, text, summary, sent_bounds, extractive_summary
"""

class ArticleDataset(Dataset):
    def __init__(self, dataset_path, predicting=False):

        #read in data into a list of dict
        lines = open(dataset_path).read().lower().strip().split('\n')
        datas = [json.loads(line) for line in lines]

        #initialize dict with SOS and EOS
        self.word2index = {'<SOS>' : 0, '<EOS>' : 1, '<PAD>': 2}
        self.index2word = {0: '<SOS>', 1: '<EOS>', 2: '<PAD>'}

        # word embedding of 'SOS' and 'EOS' and 'PAD'
        self.index2vec = [[0. for _ in range(word_vec_d)] for __ in range(3)]

        #start storing each line into class Article
        self.articles = []
        for data in datas:
            #hadling i-th data
            sys.stderr.write('reading article ' + str(data['id'])+'...\n')

            #tokenize the words
            data['text'] = nltk.word_tokenize(data['text'])
            data['summary'] = [] if predicting else nltk.word_tokenize(data['summary'])

            #creating word2index , index2word, index2vec dict
            for word in data['text'] + data['summary']:
                #if the word hasn't been added into the dict
                if word not in self.word2index:
                    #add it
                    self.word2index[word] = len(self.index2vec)
                    self.index2word[len(self.index2vec)] = word
                    try:
                        self.index2vec += [glove[word]]
                    except KeyError: #if we can't find the word in glove, add a zero vector as decode
                        self.index2vec += [[0 for _ in range(word_vec_d)]]

            #use this data to create Article class
            self.articles.append(Article(data, self.word2index, self.index2vec, predicting=predicting, idx=len(self.articles)))

        #total # of vocabulary (including 0 -> 0-vec)
        self.vocab_size = len(self.index2vec)


    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        #return the i-th article's text and extractive_gt in order to train the model
        article = self.articles[int(idx)]

        #if we're predicting, we haven no extractive_gt
        if article.summary == None:
            return article.text, idx
        else:
            return article.text, article.summary, idx

    def show(self):
        print('word2index:\n----------------')
        print(self.word2index)
        #prepare word embedding: 0th index -> 0-vec
        print('index2vec:\n----------------')
        print(self.index2vec[2])
        print('Articles:')
        for i, article in enumerate(self.articles):
            print('----------------\narticle {}:'.format(i))
            article.show()

    def word_embedding(self):
        return self.index2vec

    def article(self, article_idx):
        return self.articles[int(article_idx)]
