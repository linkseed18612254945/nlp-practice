import torch
from torch.nn import functional as F
from torch import nn
import utils
import math
import numpy as np

def attention(q, k, v, mask, dropout=None):
    score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(q.shape[-1])
    mask = mask.unsqueeze(1).squeeze(1)
    score = score.masked_fill(mask, -np.inf)
    if dropout is not None:
        score = dropout(score)
    output = torch.matmul(score, v)
    return output


class GPT(nn.Module):
    def __init__(self, input_size, vocab_size, d_model, num_head, d_ff, box_num=3, pad=1):
        super(GPT, self).__init__()
        self.pad = pad
        self.embedding = nn.Embedding(input_size, d_model)
        self.encoder = Encoder(d_model, num_head, d_ff, box_num)
        self.output = self.linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        padding_mask = (x == self.pad)
        x = self.embedding(x)
        x = self.encoder(x, padding_mask)
        output = self.output(x)
        output = self.softmax(output)
        return output


class Encoder(nn.Module):
    def __init__(self, d_model, num_head, d_ff, box_num):
        super(Encoder, self).__init__()
        self.encoder_boxes = utils.module_clone(EncoderBox(d_model, num_head, d_ff), box_num)

    def forward(self, x, mask):
        for box in self.encoder_boxes:
            x = box.forward(x, box)
        return x

class EncoderBox(nn.Module):
    def __init__(self, d_model, num_head, d_ff):
        super(EncoderBox, self).__init__()
        self.self_attn = MaskedMultiHeadSelfAttention(d_model, num_head)
        self.ff = FeedForward(d_model, d_ff)
        self.layer_norm1 = AddAndLayerNormal(d_model)
        self.layer_norm2 = AddAndLayerNormal(d_model)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.layer_norm1(x, attn_output)
        ff_output = self.ff(x)
        output = self.layer_norm2(x, ff_output)
        return output

class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super(MaskedMultiHeadSelfAttention, self).__init__()
        assert d_model % num_head == 0
        self.d_model = d_model
        self.num_head = num_head
        self.linears = utils.module_clone(nn.Linear(d_model, d_model), 4)

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        seq_length = query.shape[1]
        query = self.linears[0](query).reshape(batch_size, seq_length, self.num_head, -1).transpose(1, 2)
        key = self.linears[1](key).reshape(batch_size, seq_length, self.num_head, -1).transpose(1, 2)
        value = self.linears[2](value).reshape(batch_size, seq_length, self.num_head, -1).transpose(1, 2)
        attention_value = attention(query, key, value, mask)
        attention_value = attention_value.transpose(1, 2).reshape(batch_size, seq_length, -1)
        attention_value = self.linears[3](attention_value)
        return attention_value


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        output = self.dropout(self.linear2(F.relu(self.linear1(x))))
        return output

class AddAndLayerNormal(nn.Module):
    def __init__(self, d_model):
        super(AddAndLayerNormal, self).__init__()
        self.layer_normal = nn.LayerNorm(d_model)

    def forward(self, x, output):
        output = self.layer_normal(x + output)
        return output