import torch.nn as nn
import utils
import torch
from torch.nn import functional as F
import math

def attention(query, key, value, dropout=None):
    weights = torch.matmul(query, torch.transpose(key, 2, 3)) / math.sqrt(query.shape[2])
    weights = F.softmax(weights, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    weighted_value = torch.matmul(weights, value)
    return weighted_value, weights


class Transformer(nn.Module):
    def __init__(self, d_model, output_size, num_box=3):
        super(Transformer, self).__init__()
        self.linear = nn.Linear(d_model, output_size)


class Encoder(nn.Module):
    def __init__(self, encoder_box,  num_box):
        super(Encoder, self).__init__()
        self.encoder_layer = utils.module_clone(encoder_box, num_box)


class EncoderBox(nn.Module):
    def __init__(self, self_attn, feed_forward):
        super(EncoderBox, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward


class MultiheadSelfAttention(nn.Module):
    def __init__(self, num_head, d_model, dropout_rate=0.1):
        super(MultiheadSelfAttention, self).__init__()
        self.num_head = num_head
        self.d_k = d_model / self.num_head
        self.Linears = utils.module_clone(nn.Linear(d_model, d_model), 3)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value):
        query = self.Linears[0](query).reshape(query.shape[0], query.shape[1], self.num_head, self.d_k).transpose(1, 2)
        key = self.Linears[1](key).reshape(key.shape[0], key.shape[1], self.num_head, self.d_k).transpose(1, 2)
        value = self.Linears[2](value).reshape(value.shape[0], value.shape[1], self.num_head, self.d_k).transpose(1, 2)
        weighted_value = attention(query, key, value, self.dropout)
        weighted_value = torch.transpose(1, 2).reshape()
        return weighted_value

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        output = self.linear2(self.linear1(x))
        output = self.dropout(output)
        return output

