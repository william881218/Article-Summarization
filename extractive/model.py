import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, matrix):
        super(WordEmbedding, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(matrix.astype('float32')))
    def forward(self, input):
        with torch.no_grad():
            output = self.embedding_layer(input)
        return output

class RNNTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_size):
        super(RNNTagger, self).__init__()
        self.hidden_dim = hidden_dim
        # The GRU takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, output_size)
    def forward(self, packedSequence):
        #feed into the rnn model, get the output
        #output = model(x_packed)
        output_packed, _ = self.rnn(packedSequence)
        output_padded, output_lengths = pad_packed_sequence(output_packed, batch_first=True)
        #print('output_padded: ~~~~~~``')
        #print(output_padded)
        output_padded = self.hidden2tag(output_padded)
        return output_padded, output_lengths
