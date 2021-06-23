import jieba
import jieba.posseg as pseg
import re
import simple_utils

class TokenizeProcessor:
    """
    预处理分词引擎对象，对外提供预处理、分词、词性标注等方法
    """
    def __init__(self):
        self.tokenizer, self.search_tokenizer, self.tag_tokenizer = self.__make_tokenizer("jieba")
        
        self.stopwordset = simple_utils.read_words_from_file("resource/stopwords.txt")
        self.stopwordlist = sorted(self.stopwordset, key=lambda x: len(x), reverse=True)
        self.reply_pattern = re.compile(r"回复\S*:")
        self.forward_pattern = re.compile(r"@\S*:")
        self.user_pattern = re.compile(r'@\S*:')

        self.http_pattern = re.compile(r"(http|ftp|https)://\S*([A-z]|[0-9])")

    def cn_text_process(self, data):
        docs = []
        for seq in data:
            tokens = self.cn_sentence_process(seq)
            docs.append(tokens)
        return docs

    def cn_sentence_process(self, seq):
        seq = [w for w in seq if '\u4e00' <= w <= '\u9fff']
        tokens = ' '.join([word for word in self.tokenizer(seq) if word not in self.stopwordset]).strip()
        return tokens

    def cn_sentence_clean(self, seq, rm_stopwords=True):
        seq = re.sub(self.http_pattern, "", seq)
        if rm_stopwords:
            for stopword in self.stopwordlist:
                if len(stopword) > 1:
                    seq = seq.replace(stopword, "")
        return seq

    def cn_sentence_tokenize(self, seq, tag=False, rm_stopwords=True, use_search_cut=False, use_source=False):
        if not use_source:
            seq = re.sub(self.http_pattern, "", seq)
        new_seq = []

        if tag:
            new_tags = []
            words_with_tags = self.tag_tokenizer(seq)
            for word, tag in words_with_tags:
                if rm_stopwords and (word in self.stopwordset):
                    continue
                new_seq.append(word)
                new_tags.append(tag)
            return " ".join(new_seq), " ".join(new_tags)

        if use_search_cut:
            words = self.search_tokenizer(seq)
        else:
            words = self.tokenizer(seq)
        for word in words:
            if rm_stopwords and (word in self.stopwordset):
                continue
            new_seq.append(word)
        return " ".join(new_seq)

    def wb_doc_clean(self, doc):
        if doc is None or (not isinstance(doc, str)) or len(doc) == 0:
            return doc
        main_sentence, trace_sentence, source_sentence = self.__weibo_sentence_process(doc)
        main_sentence = self.cn_sentence_clean(main_sentence)
        trace_sentence = self.cn_sentence_clean(trace_sentence)
        source_sentence = self.cn_sentence_clean(source_sentence)
        return main_sentence, trace_sentence, source_sentence

    def wb_doc_tokenize(self, doc, use_search_cut=False, use_cut_all=False):
        if doc is None or (not isinstance(doc, str)) or len(doc) == 0:
            return "", "", ""
        main_sentence, trace_sentence, source_sentence = self.__weibo_sentence_process(doc)
        main_sentence = self.cn_sentence_tokenize(main_sentence, use_search_cut=use_search_cut)
        trace_sentence = self.cn_sentence_tokenize(trace_sentence, use_search_cut=use_search_cut)
        source_sentence = self.cn_sentence_tokenize(source_sentence, use_search_cut=use_search_cut)
        return main_sentence, trace_sentence, source_sentence

    def wb_doc_tag(self, doc):
        if doc is None or (not isinstance(doc, str)) or len(doc) == 0:
            return "", "", "", "", "", ""
        main_sentence, trace_sentence, source_sentence = self.__weibo_sentence_process(doc)
        main_sentence, main_tags = self.cn_sentence_tokenize(main_sentence, tag=True)
        trace_sentence, trace_tags = self.cn_sentence_tokenize(trace_sentence, tag=True)
        source_sentence, source_tags = self.cn_sentence_tokenize(source_sentence, tag=True)
        return main_sentence, trace_sentence, source_sentence, main_tags, trace_tags, source_tags

    def get_main_content(self, doc):
        main_sentence, trace_sentence, source_sentence = self.__weibo_sentence_process(doc)
        if len(main_sentence) == 0 and len(trace_sentence) == 0:
            return source_sentence
        else:
            return main_sentence

    def __weibo_sentence_process(self, seq):
        seq = re.sub(self.http_pattern, "", seq)
        source_sentence = ""
        trace_sentence = ""
        main_sentence = ""
        if "//" in seq:
            sentences = seq.split("//")
            source_sentence = sentences[-1]
            main_sentence = sentences[0]
            main_sentence = re.sub(self.reply_pattern, "", main_sentence)

            if len(sentences) > 2:
                trace_sentence = ' '.join(sentences[1:-1])
                trace_sentence = re.sub(self.forward_pattern, "", trace_sentence)
        else:
            source_sentence = seq
        return main_sentence, trace_sentence, source_sentence

    @staticmethod
    def __make_tokenizer(tokenize_method):
        def tokenizer(seq):
            if tokenize_method == 'jieba':
                return jieba.lcut(seq)
            else:
                return seq.split(' ')

        def search_tokenizer(seq):
            if tokenize_method == 'jieba':
                return jieba.lcut_for_search(seq)
            else:
                return seq.split(' ')

        def tag_tokenizer(seq):
            if tokenize_method == 'jieba':
                return pseg.cut(seq)
            else:
                return seq.split(' ')

        return tokenizer, search_tokenizer, tag_tokenizer
