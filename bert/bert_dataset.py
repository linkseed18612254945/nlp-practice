import torchtext
from torchtext import data


class BertDataSet(data.Dataset):
    def __init__(self, path, text_field, encoding='utf-8', **kwargs):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        tokens = [text_field.tokenize(line) for line in lines]
        examples = None
        fields = [('bert_input', text_field), ('bert_label', text_field), ('segment_input', data.LabelField()), ('is_next', data.LabelField())]
        super(BertDataSet, self).__init__(examples, fields, **kwargs)


