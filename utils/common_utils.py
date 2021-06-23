import logging
import time
from functools import wraps


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
        logging.info('function[{}] run success -- cost time {:.3f} s'.format(func.__name__, end_time - start_time))
        return res

    return time_print_wrapper


def read_words_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = f.read().splitlines()
    return words


def config2dict(config):
    """
    将config对象转换为字典方便存储
    :param config: config.Config,  配置文件对象
    :return:
    """
    config_dict = {}
    attrs = dir(config)
    for attr in attrs:
        if '__' not in attr:
            config_dict[attr] = getattr(config, attr)
    return config_dict


def harm_suggest(predict_score, harm_score=80, suspicion_score=50):
    if predict_score >= harm_score:
        return 'Harm'
    elif predict_score >= suspicion_score:
        return 'Suspicion'
    else:
        return 'Normal'


def cut_sent(words, punt_list):
    """ 自定义短句切分函数 """
    start = 0
    i = 0
    sents = []
    for word in words:
        if word in punt_list:
            sents.append(words[start:i])
            start = i + 1
            i += 1
        else:
            i += 1
    if start < len(words):
        sents.append(words[start:])
    return sents

def is_contain_all_words(content, words, not_contain=False):
    res = True
    for word in words:
        if not_contain:
            res = res and (word not in content)
        else:
            res = res and (word in content)
    return res


def sentence_have_cn_char(sentence):
    """
    判断句子中是否含有中文字符
    :param sentence:
    :return:
    """
    for s in sentence:
        if '\u4e00' <= s <= '\u9fff':
            return True
    return False


