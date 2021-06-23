import re
from utils.langconv import Converter
from bs4 import BeautifulSoup

def remove_html_tag(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text().strip()

def clean_at_content_douyin(content):
    content = content.replace("\u200b", "")
    content = re.sub('@\S+@', '@对象 @', content)
    content += ' '
    at_pattern = "@\S+ "
    content = re.sub(at_pattern, '', content).strip()
    return content

def clean_at_content(content, only_user=False):
    content = content.replace("\u200b", "")
    if only_user:
        at_pattern = "@\S+?[:：]"
        content = re.sub(at_pattern, '', content).strip()
        return content
    at_pattern = "@\S+?[:：，。,.!?！？；;]"
    content = re.sub(at_pattern, '', content).strip()
    content = re.sub('@\S+@', '@对象 @', content)
    content += ' '
    at_pattern = "@\S+ "
    content = re.sub(at_pattern, '', content).strip()
    content = content.replace('回复', '')
    return content

def clean_http_urls(content):
    content = re.sub('http://[!?./a-zA-Z0-9]*', '', content)
    return content.strip()

def replaceCharEntity(htmlstr):
    CHAR_ENTITIES={'nbsp':' ','160':' ',
                'lt':'<','60':'<',
                'gt':'>','62':'>',
                'amp':'&','38':'&',
                'quot':'"','34':'"',}

    re_charEntity=re.compile(r'&#?(?P<name>\w+);')
    sz=re_charEntity.search(htmlstr)
    while sz:
        entity=sz.group()#entity全称，如&gt;
        key=sz.group('name')#去除&;后entity,如&gt;为gt
        try:
            htmlstr=re_charEntity.sub(CHAR_ENTITIES[key],htmlstr,1)
            sz=re_charEntity.search(htmlstr)
        except KeyError:
            #以空串代替
            htmlstr=re_charEntity.sub('',htmlstr,1)
            sz=re_charEntity.search(htmlstr)
    return htmlstr

def repalce(s,re_exp,repl_string):
    return re_exp.sub(repl_string,s)

def clean_html(htmlstr):
    #先过滤CDATA
    re_cdata=re.compile('//<!\[CDATA\[[^>]*//\]\]>',re.I) #匹配CDATA
    re_script=re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>',re.I)#Script
    re_style=re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>',re.I)#style
    re_br=re.compile('<br\s*?/?>')#处理换行
    re_h=re.compile('</?\w+[^>]*>')#HTML标签
    re_comment=re.compile('<!--[^>]*-->')#HTML注释
    s=re_cdata.sub('',htmlstr)#去掉CDATA
    s=re_script.sub('',s) #去掉SCRIPT
    s=re_style.sub('',s)#去掉style
    s=re_br.sub('\n',s)#将br转换为换行
    s=re_h.sub('',s) #去掉HTML 标签
    s=re_comment.sub('',s)#去掉HTML注释
    #去掉多余的空行
    blank_line=re.compile('\n+')
    s=blank_line.sub('\n',s)
    s=replaceCharEntity(s)#替换实体
    return s

def clean_topic_content(content):
    content = re.sub('#\S+?#', '', content)
    return content.strip()

def clean_no_chinese_no_english_no_digital(content):
    if content is None or not isinstance(content, str) or len(content) == 0:
        return ""
    return ''.join([w for w in content if '\u4e00' <= w <= '\u9fff' or w.isalnum()])

def clean_no_chinese(content):
    if content is None or not isinstance(content, str) or len(content) == 0:
        return ""
    return ''.join([w for w in content if '\u4e00' <= w <= '\u9fff'])

def is_zh_char(char):
    return '\u4e00' <= char <= '\u9fff' or char in " ,.!?，。？！、/@;；:：“‘\"\'”()（）\[\]【】{}|<>《》#￥%&…^*~`·+=—_\-]"

# def is_zh_char(char):
#     return '\u4e00' <= char <= '\u9fff'

def clean_seps(content):
    if content is None or not isinstance(content, str) or len(content) == 0:
        return ""
    content = re.sub('[,.!?，。？！、/@;；:：“‘"\'”()（）\[\]【】{}|<>《》#￥%&…^*~`·+=—_\-]', '', content)
    return content

def clean_no_chinese_no_english(content):
    if content is None or not isinstance(content, str) or len(content) == 0:
        return ""
    return ''.join([w for w in content if '\u4e00' <= w <= '\u9fff' or w.isalpha()])

def simplified2traditional(sentence):
    """
    将简体中文的句子转换为繁体
    :param sentence: string, 待转换字符串
    :return: string, 装换后的字符串
    """
    sentence = Converter('zh-hant').convert(sentence)
    return sentence


def traditional2simplified(sentence):
    """
    将繁体中文的句子转换为简体
    :param sentence: string, 待转换字符串
    :return: string, 装换后的字符串
    """
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


def trans_tag_to_cn(tag):
    tag_dict = {'Ag': '形语素',
                'a': '形容词',
                'ad': '副形词',
                'an': '名形词',
                'b': '区别词',
                'c': '连词',
                'dg': '副语素',
                'd': '副词',
                'e': '叹词',
                'f': '方位词',
                'g': '语素',
                'h': '前接成分',
                'i': '成语',
                'j': '简称略语',
                'k': '后接成分',
                'l': '习用语',
                'm': '数词',
                'Ng': '名语素',
                'n': '名词',
                'nr': '人名',
                'ns': '地名',
                'nt': '机构团体',
                'nz': '其他专名',
                'o': '拟声词',
                'p': '介词',
                'q': '量词',
                'r': '代词',
                's': '处所词',
                'tg': '时语素',
                't': '时间词',
                'u': '助词',
                'vg': '动语素',
                'v': '动词',
                'vd': '副动词',
                'vn': '名动词',
                'w': '标点符号',
                'x': '非语素字',
                'y': '语气词',
                'z': '状态词',
                'un': '未知词'}
    return tag_dict.get(tag) or '未知词性'
