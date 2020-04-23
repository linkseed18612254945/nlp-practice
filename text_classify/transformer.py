import torch.nn as nn
import utils
import torch

def attention(query, key, value):
    x = torch.matmul()


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


class SelfAttention(nn.Module):
    def __init__(self, num_head, d_model):
        super(SelfAttention, self).__init__()
        self.Linears = utils.module_clone(nn.Linear(d_model, d_model), 3)

    def forward(self, query, key, value):
        query = self.Linears[0](query)
        key = self.Linears[1](key)
        value = self.Linears[2](value)



class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()

