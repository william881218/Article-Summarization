import torch
import json
import nltk
import re

#word2vec_lines = open('./glove.6B.50d.txt').read().strip().split('\n')
#for line in word2vec_lines:

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
                self.tokened_text.append(word2index[word])
                #self.tokened_text.append(word)
        self.extractive_gt = article['extractive_summary']
        #self.abstractive_gt = article['summary']
        self.abstractive_gt = [word2index[word] for word in article['summary']]
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

def read_glove():
    glove = open('glove.6B.50d.txt').read().split('\n')
    glove = [line.split(' ') for line in glove]
    return {line[0]:line[1:] for line in glove}

class ArticleDataset():
    def __init__(self, dataset_path):
        lines = open(dataset_path).read().lower().strip().split('\n')
        datas = [json.loads(line) for line in lines]
        self.articles = []
        self.word2index = {}
        self.index2vec = []
        glove = read_glove()
        for data in datas:
            data['text'] = [nltk.word_tokenize(data['text'][begin:end]) for begin, end in data['sent_bounds'] ]
            data['summary'] = [word for word in nltk.word_tokenize(data['summary'])]
            for line in data['text']:
                for word in line:
                    if word not in self.word2index:
                        self.word2index[word] = len(self.index2vec)
                        self.index2vec.append(glove[word])
            for word in data['summary']:
                if word not in self.word2index:
                    self.word2index[word] = len(self.index2vec)
                    self.index2vec.append(glove[word])
            self.articles.append(Article(data, self.word2index, self.index2vec))
        pass
    def __len__(self):
        return len(self.articles)
    def __getitem__(self, idx):
        return self.articles[int(idx)]

def text_to_token(data):
    article = data['text']
    article = [article[int(start):int(end)] for start, end in data['sent_bounds']]
    return [nltk.word_tokenize(sentence) for sentence in article]



valid_dataset = ArticleDataset('./short.jsonl')
valid_dataset[0].show()
