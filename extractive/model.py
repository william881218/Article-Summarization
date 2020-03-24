import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        output_padded = self.hidden2tag(output_padded)

        return output_padded
