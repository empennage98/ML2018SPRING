import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, batch_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(46596, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
            bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2),
        )

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda()),
                Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda()))

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        out = torch.cat((lstm_out[-1, :, :self.hidden_dim],lstm_out[0, :, self.hidden_dim:]), dim=1)
        y = self.fc(out)

        return F.log_softmax(y, dim=1)
