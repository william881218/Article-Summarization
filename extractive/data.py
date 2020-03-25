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
    def __init__(self, article, word2index, predicting=False):
        self.id = article['id']
        self.sentence_cut = []
        self.tokened_text = np.ndarray(shape=(1,0))
        self.text_size = 0
        for line in article['text']:
            self.sentence_cut.append((self.text_size, self.text_size + len(line)))
            self.text_size += len(line)
            for word in line:
                self.tokened_text = np.append(self.tokened_text, [word2index[word]])
        self.extractive_gt = None if predicting else article['extractive_summary']
        self.abstractive_gt = None if predicting else [word2index[word] for word in article['summary']]
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
    def __init__(self, dataset_path, predicting=False):
        #read in data into a list of dict
        lines = open(dataset_path).read().lower().strip().split('\n')
        datas = [json.loads(line) for line in lines]

        #initialize
        self.articles = []
        self.word2index = {}
        #prepare word embedding: 0th index -> 0-vec
        self.index2vec = [[0 for _ in range(word_vec_d)]]

        #start saving each line into class Article
        for data in datas:
            #hadling i-th data
            sys.stderr.write(str(data['id'])+'\n')

            #tokenize the text and abstractive_gt
            data['text'] = [nltk.word_tokenize(data['text'][begin:end]) for begin, end in data['sent_bounds'] ]
            if not predicting:
                data['summary'] = [word for word in nltk.word_tokenize(data['summary'])]

            #handle the exception that there is no text
            if len(data['text'][0]) == 0:
                continue

            #creating word2index dict and index2vec dict
            for line in data['text']:
                for word in line:
                    #if the word hasn't been added into the dict
                    if word not in self.word2index:
                        #add it
                        self.word2index[word] = len(self.index2vec)
                        try:
                            self.index2vec += [glove[word]]
                        except KeyError: #if we can't find the word in glove, add a random vector as decode
                            self.index2vec += [np.random.standard_normal(word_vec_d).tolist()]
            #the same procedure on abstractive_gt
            if not predicting:
                for word in data['summary']:
                    if word not in self.word2index:
                        self.word2index[word] = len(self.index2vec)
                        try:
                            self.index2vec += [glove[word]]
                        except KeyError:
                            self.index2vec += [np.random.standard_normal(word_vec_d).tolist()]
            #use this data to create Article class
            self.articles.append(Article(data, self.word2index, predicting=predicting))
        #total # of vocabulary (including 0 -> 0-vec)
        self.vocab_size = len(self.index2vec)
        self.index2vec = np.array(self.index2vec)

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        #return the i-th article's text and extractive_gt in order to train the model
        article = self.articles[int(idx)]

        if article.extractive_gt == None: #if we're predicting, we haven no extractive_gt
            return torch.from_numpy(article.tokened_text.reshape([article.text_size]).astype(dtype='int32'))

        #label tokens in extractive_gt as True, while other tokens in text to be False
        extractive_gt = np.zeros(shape=(article.text_size))
        ext_cut = article.sentence_cut[int(article.extractive_gt)]
        extractive_gt[ext_cut[0]:ext_cut[1]] = 1.

        return (torch.from_numpy(article.tokened_text.reshape([article.text_size]).astype(dtype='int32')),
                torch.from_numpy(extractive_gt))

    def word_embedding(self):
        return self.index2vec
