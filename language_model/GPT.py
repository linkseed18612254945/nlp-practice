import torch
from torch.nn import functional as F
from torch import nn
import utils

class GPT(nn.Module):
    def __init__(self):
        super(GPT, self).__init__()

    def forward(self):
        pass


class EncoderBox(nn.Module):
    def __init__(self):
        super(EncoderBox, self).__init__()

    def forward(self):
        pass

class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super(MaskedMultiHeadSelfAttention, self).__init__()
        assert d_model % num_head == 0
        self.d_model = d_model
        self.num_head = num_head
        self.linears = utils.module_clone(nn.Linear(d_model, d_model), 4)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        seq_length = query.shape[1]


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
        output = x + self.layer_normal(x)
        return output