import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
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
            if not predicting:
                input = input.to(DEVICE)
            else:
                input = input.cpu()
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

    def forward(self, input, input_lengths, hidden=None):
        #input_seqs: batch_size * max_input_len
        input = input.to(DEVICE)

        outputs, hidden = self.gru(input)

        outputs = self.linear(outputs)
        outputs = self.tanh(outputs)

        hidden = outputs[:, -1, :]
        hidden = hidden.unsqueeze(0)

        return outputs, hidden

class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, max_length, max_predict_len, teacher_forcing_ratio):
        """Define layers for a  rnn decoder"""
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.max_predict_len = max_predict_len
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.gru = nn.GRU(hidden_size + word_vec_d, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size * 2 + word_vec_d, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)  # work with NLLLoss = CrossEntropyLoss
        self.tanh = nn.Tanh()

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.energy_combine = nn.Linear(hidden_size, 1, bias=False)
        self.attn_combine = nn.Linear(hidden_size + word_vec_d, hidden_size)

    def forward_step(self, inputs, hidden, encoder_outputs, index2vec, predicting=False):
        #embedded: torch.Size([5, 1, 50])
        #hidden: torch.Size([1, 5, 128])
        #encoder_outputs: torch.Size([5, 300, 128])
        embedded = index2vec(inputs, predicting).squeeze(2)
        if predicting:
            embedded = embedded.to(DEVICE)

        repeated_hidden = hidden.permute(1, 0, 2).repeat(1, self.max_length, 1) # B * max_length * hidden_dim

        energy = self.tanh(self.attn(torch.cat((encoder_outputs, repeated_hidden), 2)))
        attn_weights = self.energy_combine(energy).squeeze(2)
        attn_weights = self.softmax(attn_weights)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs) #new context vector, B x 1 x hid dim
        gru_input = torch.cat((attn_applied, embedded), 2)


        rnn_output, rnn_hidden = self.gru(gru_input, hidden)  # S = B x T(1) x H

        hidden = hidden.permute(1, 0, 2)
        rnn_output = torch.cat((rnn_output, embedded, hidden), dim=2).squeeze(1)

        #rnn_output = rnn_output.squeeze(1)  # squeeze the time dimension

        rnn_output = self.log_softmax(self.out(rnn_output))  # S = B x O



        return rnn_output, rnn_hidden

    def forward(self, encoder_outputs, context_vector, targets=None, index2vec=None, predicting=False):
        # Prepare variable for decoder on time_step_0
        max_target_length = self.max_predict_len

        if not predicting:
            target_vars, target_lengths = targets
            max_target_length = max(target_lengths)

        batch_size = context_vector.size(1)

        # Pass the context vector
        decoder_hidden = context_vector

        decoder_outputs = Variable(torch.zeros(
            self.max_length,
            batch_size,
            self.output_size
        ))  # (time_steps, batch_size, vocab_size)

        decoder_input = Variable(torch.LongTensor([[SOS_ID] * batch_size]).view(batch_size, 1, 1))

        use_teacher_forcing = True

        if predicting:
            for t in range(max_target_length):
                decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden,
                                                                         encoder_outputs, index2vec, predicting=True)
                decoder_outputs[t] = decoder_outputs_on_t
                topv, topi = torch.topk(decoder_outputs_on_t, 1)
                decoder_input = topi.view(batch_size, 1, 1)
            return decoder_outputs, decoder_hidden

        for t in range(max_target_length):
            decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs, index2vec)

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
        decoder_outputs, decoder_hidden = self.decoder.forward(encoder_outputs, encoder_hidden, targets=targets, index2vec=index2vec)
        return decoder_outputs, decoder_hidden

    def predict(self, inputs, index2vec):
        input_vars, input_lengths = inputs
        encoder_outputs, encoder_hidden = self.encoder.forward(input_vars, input_lengths)
        decoder_outputs, decoder_hidden = self.decoder.forward(encoder_outputs, encoder_hidden, targets=None, predicting=True, index2vec=index2vec)
        return decoder_outputs, decoder_hidden
