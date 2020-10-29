import copy
import os
import random
import time
from functools import wraps
import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import datetime
import time

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def time_string_to_timestamp(time_string):
    """
    将时间字符串装换为时间戳
    :param time_string:  时间字符串，格式为TIME_FORMAT格式
    :return: float， 秒级时间戳
    """
    return time.mktime(time.strptime(time_string, TIME_FORMAT))


def time_format_change(time_string, format1, format2):
    """
    将时间字符串装换为时间戳
    :param time_string:  时间字符串，格式为TIME_FORMAT格式
    :return: float， 秒级时间戳
    """
    timestamp = time.mktime(time.strptime(time_string, format1))
    new_time_string = time.strftime(format2, time.localtime(timestamp))
    return new_time_string


def timestamp_to_time_string(timestamp):
    """
    时间戳转换为时间字符串
    :param timestamp: float, 秒级时间戳
    :return: string，时间字符串，格式为TIME_FORMAT格式
    """
    time_local = time.localtime(timestamp)
    return time.strftime(TIME_FORMAT, time_local)


def get_now_time():
    """
    获得当前时间的时间字符串
    :return: string, 时间字符串，格式为TIME_FORMAT格式
    """
    return datetime.datetime.now().strftime(TIME_FORMAT)


def gap_days(gap):
    """
    获取当前时间和指定天数之前的时间
    :param gap: int， 天数
    :return start: string， gap天前的时间
    :return end: string，当前时间
    """
    now = datetime.datetime.now()
    end = now.strftime(TIME_FORMAT)
    start = (now - datetime.timedelta(gap)).strftime(TIME_FORMAT)
    return start, end


def get_range_time(start, gap_day):
    """
    获取从起始时间算起到指定天数之后的时间
    :param start: string， 指定的起始时间
    :param gap_day: int, 向后推算天数
    :return: string， 开始时间和结束时间字符串
    """
    start_time = datetime.datetime.strptime(start, TIME_FORMAT)
    end = (start_time - datetime.timedelta(gap_day)).strftime(TIME_FORMAT)
    return start, end

if __name__ == '__main__':
    f2 = TIME_FORMAT
    f1 = '%Y/%m/%d'
    c = time_format_change('2020/5/27', f1, f2)
    print(c)


def logging_time_wrapper(func):
    """
    装饰器，装饰的函数将打印运行时间
    :param func: function， 被装饰的函数对象
    :return: 装饰后的函数
    """

    @wraps(func)
    def time_print_wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        print('function[{}] run success -- cost time {:.3f} s'.format(func.__name__, end_time - start_time))
        return res

    return time_print_wrapper


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

def matrix_evaluate(y_true, y_pred):
    tp = np.sum(np.multiply(y_true, y_pred))
    fp = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
    fn = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
    tn = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))

    acc = (tp + tn) / (tp + tn + fp + fn)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 / ((1 / p) + (1 /r))
    return tp, fp, fn, tn, acc, p, r, f1

def get_samples(test_x, true_label_name='label', predict_label_name='predict_label', sort_label_name=None, sample_num=10000):
    """
    从测试数据中抽样预测结果用于展示
    :param predict_label_name:
    :param true_label_name:
    :param test_x: DataFrame, 测试数据且包含预测结果概率和标签列
    :param sample_num: int, 抽样结果数量
    :return: p_samples: DataFrame, 预测为正样本的抽样结果
    :return: nr_samples: DataFrame, 预测为负样本的且预测错误的抽样结果
    """
    fp_samples = test_x[(test_x[predict_label_name] == 1) & (test_x[true_label_name] == 0)]
    tp_samples = test_x[(test_x[predict_label_name] == 1) & (test_x[true_label_name] == 1)]
    tn_samples = test_x[(test_x[predict_label_name] == 0) & (test_x[true_label_name] == 0)]
    fn_samples = test_x[(test_x[predict_label_name] == 0) & (test_x[true_label_name] == 1)]
    if sort_label_name is not None:
        fp_samples = fp_samples.sort_values(sort_label_name, ascending=False)
        fn_samples = fn_samples.sort_values(sort_label_name, ascending=False)
    if sample_num > 0:
        fp_samples = fp_samples[:sample_num]
        fn_samples = fn_samples[:sample_num]
    return tp_samples, fp_samples, tn_samples, fn_samples


import requests
import json

headers = {'content-type': "application/json"}


def post_dict_data(url, data_dict):
    """
    发送post请求，包含一个字典数据的json的请求体
    :param url: string， post请求地址
    :param data_dict: dict, 请求体数据
    :return: response, 请求返回结果
    """
    data_json = json.dumps(data_dict)
    response = requests.post(url, data=data_json, headers=headers)
    # response = json.loads(response)
    return response

def read_words_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = f.read().splitlines()
    return words

def post_dict_return_dict_data(url, data_dict):
    """
    发送post请求，包含一个字典数据的json的请求体
    :param url: string， post请求地址
    :param data_dict: dict, 请求体数据
    :return: response, 请求返回结果
    """
    data_json = json.dumps(data_dict)
    response = requests.post(url, data=data_json, headers=headers)

    response = json.loads(response.text)
    # print(response)
    return response

def contain_words(content, words):
    if isinstance(words, str):
        with open(words, 'r', encoding='utf-8') as f:
            words = f.read().splitlines()
    for word in words:
        if word in content:
            return True
    return False

def filter_words_by_pos(tokens, pos, use_pos=('v',)):
    if isinstance(tokens, str):
        tokens = tokens.split()
    if isinstance(pos, str):
        pos = pos.split()
    use_tokens = []
    for t, p in zip(tokens, pos):
        for u in use_pos:
            if u in p and len(t) > 1:
                use_tokens.append(t)
    return use_tokens


