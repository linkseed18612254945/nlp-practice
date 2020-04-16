import copy
from torch import nn

def module_clone(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])