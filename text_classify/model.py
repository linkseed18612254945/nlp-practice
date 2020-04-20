from torch import nn
import torch
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, input_size, seq_size, embedding_size, output_size, filter_sizes=(2, 3, 4), num_filters=3, pooling_method='max'):
        super(TextCNN, self).__init__()
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.pooling_method = pooling_method
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, kernel_size=(filter_size, embedding_size)) for filter_size in filter_sizes])
        self.activate = nn.ReLU()
        self.poolings = nn.ModuleList([nn.MaxPool2d(kernel_size=(seq_size - filter_size + 1)) for filter_size in filter_sizes]) if pooling_method =='max'\
            else nn.ModuleList([nn.AvgPool2d(kernel_size=(seq_size - filter_size + 1)) for filter_size in filter_sizes])
        self.linear = nn.Linear(len(filter_sizes) * num_filters, output_size)

    def forward(self, x):
        x = self.embedding(x)
        sequence_length = x.shape[1]
        x = x.unsqueeze(dim=1)
        conv_pooling_res = []
        for conv in self.convs:
            conved = conv(x)
            conved = self.activate(conved)
            if self.pooling_method == 'max':
                pooling = nn.MaxPool2d(kernel_size=(sequence_length - len(self.filter_sizes) + 1, 1))
            else:
                pooling = nn.AvgPool2d(kernel_size=(sequence_length - len(self.filter_sizes) + 1, 1))
            pooled = pooling(conved)
            conv_pooling_res.append(pooled)

        output = torch.cat(conv_pooling_res, dim=3)
        output = torch.reshape(output, shape=(-1, len(self.filter_sizes) * self.num_filters))
        output = self.linear(output)
        return output

class RNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)

        output, hidden = self.lstm(x)
        output = output[:, -1, :]
        output = self.linear(output)
        # output = self.softmax(output)
        return output


class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        # 初始化参数
        super().__init__()

        # embedding的作用就是将每个单词变成一个词向量
        # vocab_size=词汇表长度，embedding_dim=每个单词的维度
        # padding_idx：如果提供的话，输出遇到此下标时用零填充。这里如果遇到padding的单词就用0填充。
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # output_dim输出的维度，一个数就可以了，=1
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):  # text维度为(sent_len, 1)
        embedded = self.embedding(text)
        # text 下面会指定，为一个batch的数据
        # embedded = [sent_len, batch_size, emb_dim]
        # sen_len 一条评论的单词数
        # batch_size 一个batch有多少条评论
        # emb_dim 一个单词的维度
        # 假设[sent_len, batch_size, emb_dim] = (1000, 64, 100)
        # 则进行运算: (text: 1000, 64, 25000)*(self.embedding: 1000, 25000, 100) = (1000, 64, 100)

        # [batch_size, sent_len, emb_dim] 更换顺序
        # embedded = embedded.permute(1, 0, 2)

        # [batch_size, embedding_dim]把单词长度的维度压扁为1，并降维
        # embedded 为input_size，(embedded.shape[1], 1)) 为kernel_size
        # squeeze(1)表示删除索引为1的那个维度
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)

        # (batch_size, embedding_dim)*(embedding_dim, output_dim) = (batch_size, output_dim)
        return self.fc(pooled).squeeze(1)