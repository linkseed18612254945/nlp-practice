import torchtext
from torchtext import data


class BertDataSet(data.Dataset):
    def __init__(self, path, text_field, encoding='utf-8', **kwargs):
        examples = None
        fields = [('bert_input', text_field), ('bert_label', text_field)]
        super(BertDataSet, self).__init__(examples, fields, **kwargs)