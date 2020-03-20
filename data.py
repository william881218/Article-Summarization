import torch
import json
import nltk

#word2vec_lines = open('./glove.6B.50d.txt').read().strip().split('\n')
#for line in word2vec_lines:

class Article():
        def __init__(self, id, tokened_text, extractive_gt, abstractive_gt):
            self.id = id
            self.tokened_text = tokened_text
            self.extractive_gt = extractive_gt
            self.abstractive_gt = abstractive_gt
        def show(self):
            print('id: {}'.format(self.id))
            print('tokened_text:')
            for sentence in self.tokened_text: print(sentence, end='')
            print('extractive_gt: {}'.format(self.extractive_gt))
            print('abstractive_gt: {}'.format(self.abstractive_gt))

def text_to_token(data):
    article = data['text']
    article = [article[int(start):int(end)] for start, end in data['sent_bounds']]
    return [nltk.word_tokenize(sentence) for sentence in article]

# Read a file and split into lines
def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [json.loads(line) for line in lines]

# Read datas from file into three list: tokened_text, extractive_ground_truth and abstractive_ground_truth
def readDatas(filename):
    datas = readLines(filename)
    articles = []
    for data in datas:
        tokened_text = text_to_token(data)
        articles.append(Article(data['id'], tokened_text, data['extractive_summary'], nltk.word_tokenize(data['summary'])))
    return articles

valid_datas = readDatas('./valid.jsonl')
valid_datas[0].show()
