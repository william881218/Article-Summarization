import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from data import SOS_ID, EOS_ID, PAD_ID, word_vec_d, DEVICE
import random

context_vector_dim = 128

class Index2word(nn.Module):
    def __init__(self, index2word):
        self.index2word = index2word
    def forward(self):
        pass

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, matrix):
        super(WordEmbedding, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_layer.weight.data.copy_(torch.FloatTensor(matrix))

    def forward(self, input, predicting=False):
        with torch.no_grad():

            input = input.to(DEVICE)
            #print(input)
            output = self.embedding_layer(input.long())
        return output

class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, output_size):
        """Define layers for a rnn encoder"""
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.gru = nn.GRU(embedding_size, output_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(output_size * 2, output_size)
        self.tanh = nn.Tanh()

    def forward(self, input_seqs, input_lengths, hidden=None):
        packed = pack_padded_sequence(input_seqs, input_lengths, batch_first=True)
        packed = packed.to(DEVICE)
        packed_outputs, hidden = self.gru(packed)
        batch_size = hidden.size(1)

        hidden = hidden.transpose(0, 1).contiguous().reshape(1, batch_size, -1)
        hidden = self.linear(hidden)
        hidden = self.tanh(hidden)

        return packed_outputs, hidden

class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, max_length, teacher_forcing_ratio):
        """Define layers for a  rnn decoder"""
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(word_vec_d, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)  # work with NLLLoss = CrossEntropyLoss

        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward_step(self, inputs, hidden, index2vec, predicting=False):

        embedded = index2vec(inputs, predicting)
        embedded = embedded.squeeze(2)
        if predicting:
            embedded = embedded.to(DEVICE)

        rnn_output, hidden = self.gru(embedded, hidden)  # S = T(1) x B x H
        rnn_output = rnn_output.squeeze(1)  # squeeze the time dimension
        output = self.softmax(self.out(rnn_output))  # S = B x O

        return output, hidden

    def forward(self, context_vector, targets=None, index2vec=None, predicting=False):
        # Prepare variable for decoder on time_step_0
        max_target_length = self.max_length
        batch_size = context_vector.size(1)

        if not predicting:
            target_vars, target_lengths = targets
            max_target_length = max(target_lengths)

        # Pass the context vector
        decoder_hidden = context_vector

        decoder_outputs = Variable(torch.zeros(
            max_target_length,
            batch_size,
            self.output_size
        ))  # (time_steps, batch_size, vocab_size)

        #Prepare decoder input: [SOS_ID * batchsize]
        decoder_input = Variable(torch.LongTensor([[SOS_ID] * batch_size]).view(batch_size, 1, 1))

        use_teacher_forcing = True

        if predicting:
            for t in range(max_target_length):
                #Feed inputs into decoder and get the outputs
                decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, index2vec, predicting=True)
                decoder_outputs[t] = decoder_outputs_on_t

                #Choose the most possible word and make it the next decoder input
                topv, topi = torch.topk(decoder_outputs_on_t, 1)
                decoder_input = topi.view(batch_size, 1, 1)
            return decoder_outputs, decoder_hidden

        for t in range(max_target_length):
            decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, index2vec)
            decoder_outputs[t] = decoder_outputs_on_t
            if use_teacher_forcing:
                decoder_input = target_vars[:, t:t+1].view(batch_size, 1, 1)
            else:
                topv, topi = torch.topk(decoder_outputs_on_t, 1)
                decoder_input = topi.view(batch_size, 1, 1)

        return decoder_outputs, decoder_hidden

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs, targets, index2vec):
        input_vars, input_lengths = inputs
        encoder_outputs, encoder_hidden = self.encoder.forward(input_vars, input_lengths)
        decoder_outputs, decoder_hidden = self.decoder.forward(context_vector=encoder_hidden, targets=targets, index2vec=index2vec)
        return decoder_outputs, decoder_hidden

    def predict(self, inputs, index2vec):
        input_vars, input_lengths = inputs
        encoder_outputs, encoder_hidden = self.encoder.forward(input_vars, input_lengths)
        decoder_outputs, decoder_hidden = self.decoder.forward(context_vector=encoder_hidden, targets=None, predicting=True, index2vec=index2vec)
        return decoder_outputs, decoder_hidden
