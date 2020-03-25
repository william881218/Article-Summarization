import torch
from data import *
from model import *
import random
import time
import math
from torch.nn.utils.rnn import pad_sequence
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import one_hot

batch_size = 5
training_data_path = './medium.jsonl'
learning_rate = 0.05
rnn_hidden_dim=64
tag_type_num=2
ipoch_num = 10
all_losses = []



def pad_collate(batch):
    x_data, y_data = zip(*batch)
    x_data = list(x_data)
    y_data = list(y_data)
    x_data.sort(key=lambda x: len(x), reverse=True)
    y_data.sort(key=lambda x: len(x), reverse=True)
    x_len = [len(x) for x in x_data]
    y_len = [len(y) for y in y_data]
    x_padded = pad_sequence(x_data, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y_data, batch_first=True, padding_value=0)
    return x_padded, y_padded, x_len, y_len

def main():


    #prepare training dataset with DataLoader, which will return a batched, padded sample
    training_dataset = ArticleDataset(training_data_path)
    article_data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    #prepare embedding layer
    embedding = WordEmbedding(training_dataset.vocab_size, word_vec_d, training_dataset.index2vec)

    #creating model
    rnn = RNNTagger(embedding_dim=word_vec_d, hidden_dim=rnn_hidden_dim, output_size=tag_type_num)
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
    criterion = BCEWithLogitsLoss()

    if torch.cuda.is_available():
        rnn = rnn.cuda()
        embedding = embedding.cuda()


    #training start
    for i_ipoch in range(ipoch_num):
        for i_batch, (x_padded, y_padded, x_lens, y_lens) in enumerate(article_data_loader):
            x_padded = x_padded.long()
            if torch.cuda.is_available():
                x_padded = x_padded.cuda()
                y_padded = y_padded.cuda()
            #Convert word to vector
            x_embed = embedding(x_padded)
            x_embed = Variable(x_embed)

            #packing
            x_packed = pack_padded_sequence(x_embed, x_lens, batch_first=True, enforce_sorted=False)



            #Feeding into rnn and get an output
            output, output_lengths = rnn(x_packed)
            y_padded = one_hot(y_padded.long(), 2).type_as(output)

            #evaluate
            #clear the gradient
            optimizer.zero_grad()
            loss_sum = 0

            #calculate loss acording to each sentence's length
            for i_sent, sent_len in enumerate(output_lengths):
                loss = criterion(output[i_sent, 0:sent_len], y_padded[i_sent, 0:sent_len])
                #
                loss = loss / batch_size
                loss_sum += loss.data.item()
                loss.backward(retain_graph= False if i_sent == batch_size - 1 else True)

            optimizer.step()


            all_losses.append(loss_sum / batch_size)
            if i_batch % 1 == 0:
                print('ipoch {}, batch {}, current loss: {}'.format(i_ipoch, i_batch, all_losses[-1]))
    #training end
    torch.save(rnn, 'ipoch={}_lr={}_data={}.pt'.format(ipoch_num, learning_rate, training_data_path[2:]))
    print('Losses:\n----------------')
    for i in all_losses:
        print(i)



if __name__ == '__main__':
    main()
