import copy
import os
import random

import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def remove_str_from_sentence(text, strs):
    for s in strs:
        text = text.replace(s, '')
    return text

def module_clone(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def weight_init(net, exclude_layer=('lstm'), weight_method='xavier', bias_constant=0):
    for name, parameter in net.named_parameters():
        layer_name = name.split('.')[0]
        parameter_type = name.split('.')[1]
        if True:
            if layer_name in exclude_layer:
                continue
            if parameter_type == 'weight':
                if weight_method == 'xavier':
                    nn.init.xavier_normal_(parameter)
                elif weight_method == 'kaiming':
                    nn.init.kaiming_normal_(parameter)
            elif parameter_type == 'bias':
                nn.init.constant_(parameter, bias_constant)


def evaluate_report(y_true, y_pred, labels=None, target_names=None):
    return classification_report(y_true, y_pred, labels, target_names)

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average='micro')
    r = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    return acc, p, r, f1

def seed_torch(seed=2018):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True