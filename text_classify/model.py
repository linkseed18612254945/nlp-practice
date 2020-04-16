from torch import nn
import torch


class RNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers)
        self.linear = nn.Linear(embedding_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        output, _ = self.lstm(x)
        output = output.view(-1, self.hidden_size)
        output = self.linear(output)
        output = self.softmax(output)
        return output


class DNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers=3, dropout=0.1):
        super(DNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.linear_first = nn.Linear(embedding_size, hidden_size)
        self.linear_middle = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.linear_output = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        # x = self.dropout(x)
        x = torch.sum(x, dim=1)
        x = self.linear_first(x)
        for l in self.linear_middle:
            x = l(x)
        x = self.linear_output(x)
        x = self.softmax(x)
        return x