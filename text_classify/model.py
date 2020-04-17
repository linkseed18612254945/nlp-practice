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
        output = output[:, -1, :]
        output = self.linear(output)
        output = self.softmax(output)
        return output


class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        # 初始化参数
        super().__init__()

        # embedding的作用就是将每个单词变成一个词向量
        # vocab_size=词汇表长度，embedding_dim=每个单词的维度
        # padding_idx：如果提供的话，输出遇到此下标时用零填充。这里如果遇到padding的单词就用0填充。
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

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
        embedded = embedded.permute(1, 0, 2)

        # [batch_size, embedding_dim]把单词长度的维度压扁为1，并降维
        # embedded 为input_size，(embedded.shape[1], 1)) 为kernel_size
        # squeeze(1)表示删除索引为1的那个维度
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)

        # (batch_size, embedding_dim)*(embedding_dim, output_dim) = (batch_size, output_dim)
        return self.fc(pooled)