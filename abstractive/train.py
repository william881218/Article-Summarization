import torch
from data import *
from model import *
import random
import time
import math
from torch.nn.utils.rnn import pad_sequence
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import one_hot

TRAINING_DATASET_PATH = '../valid_short.jsonl'
VALIDATION_DATASET_PATH = '../valid_short.jsonl'
batch_size = 10
validation_batch_size = 20
learning_rate = 0.1
using_adam = True
adam_learning_rate = 0.01
SGD_momentum = 0.8
rnn_hidden_dim=64
tag_type_num=1
ipoch_num = 50
max_output_len = 300
teacher_forcing_ratio = 0.5
all_losses = []


def pad_collate(batch):
    #sort the batch according to sentence's length (requested by padding)
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    #Calculate each sample's length
    x_data, y_data, idx = zip(*batch)
    x_data = list(x_data)
    y_data = list(y_data)
    x_len = [len(x) for x in x_data]
    y_len = [len(y) for y in y_data]

    #Pad each label and calculate each sentence's length
    x_padded = pad_sequence(x_data, batch_first=True, padding_value=PAD_ID)
    y_padded = pad_sequence(y_data, batch_first=True, padding_value=PAD_ID)
    return x_padded, y_padded.long(), x_len, y_len, list(idx)

#Used to generate validation DataLoader iterator
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train(seq2seq, batch, criterion, training_embedding):
    #Unzip a batch
    x_padded, y_padded, x_len, y_len, idx = batch

    #Feed into model, get the output of shape (Timestamp x Batch size x Output space)
    decoder_output, hidden = seq2seq((x_padded, x_len), (y_padded, y_len), training_embedding)

    #Flatten the output to calculate Loss only once
    t, b = decoder_output.size(0), decoder_output.size(1)
    decoder_output = decoder_output.reshape(b * t, -1)

    y_padded = y_padded.transpose(0, 1).contiguous().view(-1) #Flatten the taret label

    return criterion(decoder_output, y_padded) #return Loss

def main(training_dataset_path=TRAINING_DATASET_PATH, validation_dataset_path=VALIDATION_DATASET_PATH, model_path=None):

    #prepare training dataset with DataLoader, which will return a batched, padded sample
    training_dataset = ArticleDataset(training_dataset_path)
    training_data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    training_dataLoader_it = iter(cycle(training_data_loader))

    #prepare validation dataset
    validation_dataset = ArticleDataset(validation_dataset_path)
    valid_data_loader = DataLoader(validation_dataset, batch_size = validation_batch_size, shuffle=True, collate_fn=pad_collate)
    valid_dataLoader_it = iter(cycle(valid_data_loader))

    #Create model
    encoder = Encoder(vocab_size=training_dataset.vocab_size, embedding_size=word_vec_d, output_size=context_vector_dim)
    decoder = Decoder(hidden_size=context_vector_dim,
                      output_size=training_dataset.vocab_size,
                      max_length=max_output_len,
                      teacher_forcing_ratio=teacher_forcing_ratio)
    seq2seq = Seq2Seq(encoder, decoder)

    #Create and save the WordEmbedding, which will be used in decoder
    training_embedding = WordEmbedding(vocab_size=training_dataset.vocab_size,
                                       embedding_dim=word_vec_d, matrix=training_dataset.index2vec)
    torch.save(training_embedding, '{}/index2vec.pt'.format(model_path))

    #Save the index2word dict in order to translate the outputs to words in predict.py
    training_index2word = Index2word(training_dataset.index2word)
    torch.save(training_index2word, '{}/index2word.pt'.format(model_path))

    #Choose optimizer and criterion
    optimizer = torch.optim.SGD(seq2seq.parameters(), lr=learning_rate, momentum=SGD_momentum)
    if using_adam:
        optimizer = torch.optim.Adam(seq2seq.parameters(), lr=adam_learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_ID)

    #Calculate on GPU if possible
    seq2seq.to(DEVICE)
    criterion.to(DEVICE)
    training_embedding.to(DEVICE)

    #training start
    for i_ipoch in range(ipoch_num):
        for i_batch, batch in enumerate(training_data_loader):
            #Zero the grads, calculate loss and backpropagate
            optimizer.zero_grad()
            cur_loss = train(seq2seq, batch, criterion, training_embedding)
            cur_loss.backward()
            optimizer.step()

            #record losses
            all_losses.append(cur_loss.item())

            #record the losses and print the info every 5 batch
            if i_batch % 5 == 0:
                with torch.no_grad():
                    #Validation
                    valid_loss = train(seq2seq, next(valid_dataLoader_it), criterion, training_embedding)
                    print('ipoch [{}/{}], step [{}/{}], loss: {}, valid_loss: {}'.format(i_ipoch+1,
                      ipoch_num, i_batch + 1, len(training_dataset) // batch_size + 1, all_losses[-1], valid_loss.item()))

                    #Save the model every 200 batches
                    if i_batch % 200 == 0:
                        torch.save(seq2seq, '{}/ipoch={}_ibatch={}_loss={}_valid={}.pt'.format(
                                            model_path, i_ipoch, i_batch, all_losses[-1], valid_loss.item()))
    #training ends
    return




if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print('usage: python train.py [training dataset path] [validation dataset path] [model directory path]')
        print('if path not assigned, training dataset: {}, validation dataset: {}'.format(TRAINING_DATASET_PATH, VALIDATION_validation_batch_size))
        exit()
