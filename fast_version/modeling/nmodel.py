import torch.nn as nn
from torch.autograd import Variable
from linear_nce import linear_nce


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, emb_dim, nhid, nlayers, loss_type, dropout=0.5, unigram_prob=None, num_noise=64):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(emb_dim, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(emb_dim, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.loss_type = loss_type
        self.num_noise = num_noise
        if loss_type == 'nce':
            self.decoder = linear_nce(nhid, ntoken, unigram_prob)
        else:
            self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.mode = 'train'

    def set_mode(self, m):
        self.mode = m

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, target=None):
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(input, hidden)
        output = self.drop(output)
        if self.loss_type == 'nce':
            if self.mode == 'eval':
                decoded = self.decoder(output.view(output.size(
                    0) * output.size(1), output.size(2)), mode='eval_full')
            else:
                decoded = self.decoder(output.view(output.size(
                    0) * output.size(1), output.size(2)), target, mode='train', num_noise=self.num_noise)
            # print(decoded)
            return decoded, hidden
        else:
            decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
            return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())