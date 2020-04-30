import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=1, dropout=0.2):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = self.dropout(x)
        output, hidden = self.lstm(x, hidden)
        output = output.view(-1, self.hidden_size)
        output = self.linear(output)
        output = self.softmax(output)
        return output, hidden

class CS224NRnnModel(nn.Module):
    def __init__(self, vocab_size, config):
        super(CS224NRnnModel, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config['embedding_size'])
        self.rnn = nn.RNN(config['embedding_size'], config['hidden_size'], config['hidden_layers'], batch_first=True)
        self.output = nn.Linear(config['hidden_size'], vocab_size)
        self.dropout = nn.Dropout(p=config['dropout_rate'])
        self.softmax = nn.LogSoftmax()
        self.init_weight()

    def init_weight(self):
        self.embedding.weight = nn.init.xavier_uniform(self.embedding.weight)
        self.output.weight = nn.init.xavier_uniform(self.output.weight)
        self.output.bias.data.fill_(0)

    def init_hidden(self):
        hidden = torch.zeros(self.config['hidden_layers'], self.config['batch_size'], self.config['hidden_size'])
        return hidden

    def forward(self, input_x, hidden):
        input_embedding = self.embedding(input_x)    # B x S x D
        input_embedding = self.dropout(input_embedding)
        output, rnn_hidden = self.rnn(input_embedding, hidden)  # hidden: L x B x H , output: B x S x H
        output = output.contiguous().view(output.size(0) * output.size(1), -1).contiguous()  # (B * S) x H
        output_prob = self.output(output)  # (B * S) x V
        last_output = self.output(rnn_hidden[self.config['hidden_layers'] - 1, :, :])  # B x V
        return output_prob, rnn_hidden, last_output