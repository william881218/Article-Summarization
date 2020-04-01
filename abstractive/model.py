import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from data import SOS_ID, EOS_ID, PAD_ID, word_vec_d, GPU_DEVICE
import random

context_vector_dim = 128

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, matrix):
        super(WordEmbedding, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_layer.weight.data.copy_(torch.FloatTensor(matrix))

    def forward(self, input):
        with torch.no_grad():
            output = self.embedding_layer(input.long())
        return output


class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, output_size):
        """Define layers for a rnn encoder"""
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.gru = nn.GRU(embedding_size, output_size, batch_first=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        packed = pack_padded_sequence(input_seqs, input_lengths, batch_first=True)
        packed_outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs, hidden

class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, max_length, teacher_forcing_ratio, index2vec):
        """Define layers for a  rnn decoder"""
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(word_vec_d, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)  # work with NLLLoss = CrossEntropyLoss

        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.index2vec = index2vec

    def forward_step(self, inputs, hidden):
        # inputs: (time_steps=1, batch_size)
        #batch_size = inputs.size(0)
        embedded = self.index2vec(inputs).squeeze(2)
        #print(embedded.shape)
        rnn_output, hidden = self.gru(embedded, hidden)  # S = T(1) x B x H
        #print('rnn output')
        #print(rnn_output)
        #print(rnn_output.shape)
        rnn_output = rnn_output.squeeze(1)  # squeeze the time dimension
        #print('rnn output after squeeze')
        #print(rnn_output)
        output = self.log_softmax(self.out(rnn_output))  # S = B x O
        #print('rnn output LogSoftmaxed')
        #print(output)
        #print(output.shape)
        return output, hidden

    def forward(self, context_vector, targets=None, predicting=False):
        # Prepare variable for decoder on time_step_0
        max_target_length = self.max_length

        if not predicting:
            target_vars, target_lengths = targets
            max_target_length = max(target_lengths)

        batch_size = context_vector.size(1)

        # Pass the context vector
        decoder_hidden = context_vector

        decoder_outputs = Variable(torch.zeros(
            max_target_length,
            batch_size,
            self.output_size
        ))  # (time_steps, batch_size, vocab_size)

        decoder_input = Variable(torch.LongTensor([[SOS_ID] * batch_size]).view(batch_size, 1, 1))

        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda(GPU_DEVICE)
            decoder_outputs = decoder_outputs.cuda(GPU_DEVICE)

        use_teacher_forcing = True if random.random() > self.teacher_forcing_ratio else False

        if predicting:
            for t in range(self.max_length):
                decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                decoder_outputs[t] = decoder_outputs_on_t
                topv, topi = torch.topk(decoder_outputs_on_t, 1)
                decoder_input = topi.view(batch_size, 1, 1)
            return decoder_outputs, decoder_hidden


        for t in range(max_target_length):
            decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            #print(decoder_outputs_on_t)
            #print(decoder_outputs_on_t.shape)

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

    def forward(self, inputs, targets):
        input_vars, input_lengths = inputs
        encoder_outputs, encoder_hidden = self.encoder.forward(input_vars, input_lengths)
        decoder_outputs, decoder_hidden = self.decoder.forward(context_vector=encoder_hidden, targets=targets)
        return decoder_outputs, decoder_hidden

    def predict(self, inputs):
        input_vars, input_lengths = inputs
        encoder_outputs, encoder_hidden = self.encoder.forward(input_vars, input_lengths)
        decoder_outputs, decoder_hidden = self.decoder.forward(context_vector=encoder_hidden, targets=None, predicting=True)
        return decoder_outputs, decoder_hidden

