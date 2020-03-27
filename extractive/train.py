import torch
from data import *
from model import *
import random
import time
import math
from torch.nn.utils.rnn import pad_sequence
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import one_hot

TRAINING_DATASET_PATH = './train.jsonl'
VALIDATION_DATASET_PATH = './valid.jsonl'
batch_size = 16
validation_batch_size = 50
learning_rate = 0.1
using_adam = True
adam_learning_rate = 0.01
SGD_momentum = 0.8
rnn_hidden_dim=64
tag_type_num=1
ipoch_num = 50
all_losses = []

def pad_collate(batch):
    #sort the batch according to sentence's length (requested by padding)
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    #Pad each label and calculate each sentence's length
    x_data, y_data, idx = zip(*batch)
    x_data = list(x_data)
    y_data = list(y_data)
    x_len = [len(x) for x in x_data]
    y_len = [len(y) for y in y_data]
    x_padded = pad_sequence(x_data, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y_data, batch_first=True, padding_value=0)
    return x_padded.long(), y_padded, x_len, y_len, idx

def validate(rnn, valid_dataLoader_it, validation_embedding, validation_dataset):
    with torch.no_grad():
        x_padded, y_padded, x_len, y_len, idx = next(valid_dataLoader_it)
        if torch.cuda.is_available():
            x_padded = x_padded.cuda(GPU_DEVICE)
            y_padded = y_padded.cuda(GPU_DEVICE)

        #Convert word to vector
        x_embed = validation_embedding(x_padded)

        #packing
        x_packed = pack_padded_sequence(x_embed, x_len, batch_first=True, enforce_sorted=False)

        #Feeding into rnn and get an output
        output, output_lengths = rnn(x_packed)
        output = output.view(output.shape[0], -1)

        #Calculate accuracy
        acc_count = 0
        for i_sent, sent_len in enumerate(output_lengths):
            extractive_gt = validation_dataset.article(idx[i_sent]).extractive_gt
            predicted_gt = validation_dataset.predict(idx[i_sent], output[i_sent, 0:sent_len])
            acc_count += (predicted_gt == extractive_gt)

        return acc_count / len(idx), len(idx)

#Used to generate validation DataLoader iterator
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def main(training_dataset_path=TRAINING_DATASET_PATH, validation_dataset_path=VALIDATION_DATASET_PATH):

    #prepare training dataset with DataLoader, which will return a batched, padded sample
    training_dataset = ArticleDataset(training_dataset_path)
    training_data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    #prepare validation dataset
    validation_dataset = ArticleDataset(validation_dataset_path)
    valid_data_loader = DataLoader(validation_dataset, batch_size = validation_batch_size, shuffle=True, collate_fn=pad_collate)
    valid_dataLoader_it = iter(cycle(valid_data_loader))

    #prepare embedding layer
    trainingSet_embedding = WordEmbedding(training_dataset.vocab_size, word_vec_d, training_dataset.index2vec)
    validation_embedding = WordEmbedding(validation_dataset.vocab_size, word_vec_d, validation_dataset.index2vec)

    #create model
    rnn = RNNTagger(embedding_dim=word_vec_d, hidden_dim=rnn_hidden_dim, output_size=tag_type_num)

    #Choose optimizer
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate, momentum=SGD_momentum)
    if using_adam:
        optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

    #use pos_weight in BCELoss to handle data imbalance
    pos_tag = training_dataset.positive_tag
    neg_tag = training_dataset.total_tag - pos_tag
    print('pos:{}, neg:{}'.format(pos_tag, neg_tag))
    criterion = BCEWithLogitsLoss(pos_weight=torch.FloatTensor([neg_tag / pos_tag]))

    if torch.cuda.is_available():
        rnn = rnn.cuda(GPU_DEVICE)
        trainingSet_embedding = trainingSet_embedding.cuda(GPU_DEVICE)
        validation_embedding.cuda(GPU_DEVICE)
        criterion.cuda(GPU_DEVICE)

    #training start
    for i_ipoch in range(ipoch_num):
        for i_batch, (x_padded, y_padded, x_lens, y_lens, sent_idx) in enumerate(training_data_loader):
            if torch.cuda.is_available():
                x_padded = x_padded.cuda(GPU_DEVICE)
                y_padded = y_padded.cuda(GPU_DEVICE)

            #Convert word to vector
            x_embed = trainingSet_embedding(x_padded)
            x_embed = Variable(x_embed)

            #packing
            x_packed = pack_padded_sequence(x_embed, x_lens, batch_first=True, enforce_sorted=False)

            #Feeding into rnn and get an output
            output, output_lengths = rnn(x_packed)
            output = output.view(output.shape[0], -1)
            y_padded = y_padded.type_as(output)

            #evaluate start
            loss_sum = 0
            #calculate loss acording to each sentence's length
            for i_sent, sent_len in enumerate(output_lengths):
                optimizer.zero_grad()
                loss = criterion(output[i_sent, 0:sent_len], y_padded[i_sent, 0:sent_len])
                loss_sum += loss.data.item()
                loss.backward(retain_graph=True)
                optimizer.step()

            #record the losses and print the info every 10 batch
            all_losses.append(loss_sum)
            if i_batch % 10 == 0:
                valid_acc, valid_num = validate(rnn, valid_dataLoader_it, validation_embedding, validation_dataset)
                print('ipoch [{}/{}], step [{}/{}], current loss: {}, size {}, valid_acc: {}, size: {}'.format(i_ipoch+1,
                      ipoch_num, i_batch + 1, len(training_dataset) // batch_size + 1, all_losses[-1], len(output_lengths), valid_acc, valid_num))

    #training end, save the model
    torch.save(rnn, 'ipoch={}_lr={}_train={}_opt={}.pt'.format(ipoch_num, learning_rate, training_dataset_path[2:], 'Adam' if using_adam else 'SGD'))
    return

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print('usage: python train.py [training dataset path] [validation dataset path]')
        print('if path not assigned, training dataset: {}, validation dataset: {}'.format(TRAINING_DATASET_PATH, VALIDATION_validation_batch_size))
        exit()
