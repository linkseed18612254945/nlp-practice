from utils import web_utils
import json
import pypinyin


def get_keywords_by_filter_res(res):
    keywords = []
    if isinstance(res, str):
        res = res.replace(r'\\"', "")
        res = json.loads(res).get("result")
    else:
        res = res.get("result")
    if res is None:
        return None
    res = res.get("descs")
    for rob in res:
        word = rob.get("hit_black")
        keywords.append(word)
    return keywords


def find_keywords(kw_service_url, sys_code, text):
    filter_param = {
        "id": "test",
        "text": text,
        "sys_code": sys_code
    }
    res = web_utils.post_dict_return_dict_data(kw_service_url, filter_param)
    kw = get_keywords_by_filter_res(res)
    return kw


def check_contains_ldr_kw(content, kws):
    for kw in kws:
        if kw in content:
            return True
    return False

def contain_words(content, words):
    if isinstance(words, str):
        with open(words, 'r', encoding='utf-8') as f:
            words = f.read().splitlines()
    for word in words:
        if word in content:
            return True
    return False

def check_contains_kw(content, kws):
    if isinstance(kws, str):
        with open(kws, 'r', encoding='utf-8') as f:
            kws = f.read().splitlines()
    for kw in kws:
        if kw in content:
            return kw
    return None


def check_pinyin(content, kw):
    kw_pinyin = pypinyin.lazy_pinyin(kw)
    content_pinyin = pypinyin.lazy_pinyin(content)
    return ' '.join(kw_pinyin) in ' '.join(content_pinyin)
