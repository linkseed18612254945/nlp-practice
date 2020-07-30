import torch
from torch import nn
import torchvision

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = torchvision.models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        pass
