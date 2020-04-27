import torch.nn as nn
import utils
import torch
from torch.nn import functional as F
import math

def attention(query, key, value, dropout=None):
    weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(query.shape[3])
    weights = F.softmax(weights, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    weighted_value = torch.matmul(weights, value)
    return weighted_value, weights


class Transformer(nn.Module):
    def __init__(self, input_size, d_model, num_head, d_ff, output_size, num_box=4):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(input_size, d_model)
        self.register_buffer('pe',  self.build_position_embedding(d_model))
        encoder_box = EncoderBox(MultiheadSelfAttention(d_model, num_head), FeedForward(d_model, d_ff))
        self.encoder = Encoder(encoder_box, num_box)
        self.output = nn.Linear(d_model, output_size)

        self.softmax = nn.LogSoftmax(dim=-1)

    def build_position_embedding(self, d_model, max_len=5000):
        position_embedding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        position_embedding[:, 0::2] = torch.sin(position * div_term)
        position_embedding[:, 1::2] = torch.cos(position * div_term)
        position_embedding = position_embedding.unsqueeze(0)
        return position_embedding

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pe[:, :x.size(1)]
        x = self.encoder(x)
        output = self.output(x)
        output = F.avg_pool2d(output, kernel_size=(output.shape[1], 1)).squeeze(1)
        output = self.softmax(output)
        return output


class Encoder(nn.Module):
    def __init__(self, encoder_box,  num_box=4):
        super(Encoder, self).__init__()
        self.encoder_layer = utils.module_clone(encoder_box, num_box)

    def forward(self, x):
        for encoder_box in self.encoder_layer:
            x = encoder_box(x)
        return x


class EncoderBox(nn.Module):
    def __init__(self, self_attn, feed_forward):
        super(EncoderBox, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.layer_norm1 = AddAndLayerNorm(self_attn.d_model)
        self.layer_norm2 = AddAndLayerNorm(self_attn.d_model)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.layer_norm1(x, attn_output)
        ff_output = self.feed_forward(x)
        output = self.layer_norm2(x, ff_output)
        return output


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout_rate=0.1):
        super(MultiheadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_head = num_head
        assert d_model % num_head == 0
        self.d_k = d_model / self.num_head
        self.Linears = utils.module_clone(nn.Linear(d_model, d_model), 3)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        query = self.Linears[0](query).reshape(batch_size, seq_len, self.num_head, -1).transpose(1, 2)
        key = self.Linears[1](key).reshape(batch_size, seq_len, self.num_head, -1).transpose(1, 2)
        value = self.Linears[2](value).reshape(batch_size, seq_len, self.num_head, -1).transpose(1, 2)
        weighted_value, attn = attention(query, key, value, self.dropout)
        weighted_value = weighted_value.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return weighted_value


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        output = self.linear2(F.relu(self.linear1(x)))
        output = self.dropout(output)
        return output

class AddAndLayerNorm(nn.Module):
    def __init__(self, d_model):
        super(AddAndLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, output):
        output = x + self.norm(output)
        return output

if __name__ == '__main__':
    model = Transformer(input_size=100, d_model=512, num_head=4, d_ff=128, output_size=2)