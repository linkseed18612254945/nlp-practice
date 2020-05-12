import torchtext
from torchtext import data
from torchtext.data import Iterator, Example
from random import randrange, shuffle, random, choice

class BertIterator(Iterator):

    def __init__(self, max_mask=5, **kwargs):
        super(BertIterator, self).__init__(**kwargs)
        self.max_mask = max_mask
        self.label = data.LabelField()

    def data(self):
        if self.sort:
            xs = sorted(self.dataset, key=self.sort_key)
        elif self.shuffle:
            xs = [self.dataset[i] for i in self.random_shuffler(range(len(self.dataset)))]
        else:
            xs = self.dataset
        examples = []
        for i in range(len(xs) - 1):
            random_next = randrange(len(xs))
            sentence1 = xs[i].text
            sentence2 = xs[i + 1].text
            sentence3 = xs[random_next].text
            examples.append(self.build_bert_example(sentence1, sentence2, 1))
            examples.append(self.build_bert_example(sentence1, sentence3, 0))
        return torchtext.data.dataset.Dataset(examples, fields=[('text', TEXT), ('segment', self.label),
                                        ('mask_position', self.label), ('is_next', self.label)])

    def build_bert_example(self, sentence1, sentence2, is_next):
        TEXT = self.dataset.fields['text']
        bert_input = ['<cls>'] + sentence1 + ['<sep>'] + sentence2 + ['<sep>']
        segment_input = [0] * (1 + len(sentence1) + 1) + [1] * (len(sentence2) + 1)
        mask_num = min(self.max_mask, max(1, int(len(bert_input) * 0.15)))
        can_mask_index = [i for i, token in enumerate(bert_input) if token != '<cls>' and token != '<sep']
        can_mask_index = shuffle(can_mask_index)
        mask_index = can_mask_index[:mask_num]
        for index in mask_index:
            rnd = random()
            if rnd < 0.8:
                bert_input[index] = '<mask>'
            elif rnd < 0.9:
                bert_input[index] = choice(TEXT.vocab.itos)
            else:
                pass
        if len(mask_index) < self.max_mask:
            mask_index += [0] * (self.max_mask - len(mask_index))

        return Example.fromlist(data=[bert_input, segment_input, mask_index, is_next],
                                fields=[('text', TEXT), ('segment', self.label),
                                        ('mask_position', self.label), ('is_next', self.label)])

if __name__ == '__main__':
    path = '/home/ubuntu/likun/nlp_data/bert/ptb_sentence.csv'
    TEXT = torchtext.data.Field(unk_token='<unk>', use_vocab=True)
    dataset = data.TabularDataset(path, format='csv', fields=[('text', TEXT)], skip_header=True)
    TEXT.build_vocab(dataset)
    bert_iterator = BertIterator()
    d = bert_iterator.data()
    print(dataset)