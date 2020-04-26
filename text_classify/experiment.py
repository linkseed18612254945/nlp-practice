import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torchtext import data, datasets
from torchtext.vocab import Vectors
import spacy
from tqdm import tqdm
from spacy.symbols import ORTH
from spacy.lang.en.stop_words import STOP_WORDS
import utils

USE_GPU = True
GPU_INDEX = 0
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(GPU_INDEX))
else:
    device = torch.device('cpu')

# DATA_BASE_PATH = '../nlp_data'  # Test env data path
DATA_BASE_PATH = '/home/ubuntu/likun/nlp_data'
# DATA_DIR = 'text_classify/aclImdb'
# DATA_DIR = 'text_classify/car_comments'
DATA_DIR = 'text_classify/zh_news'
DATA_TRAIN_FILE_NAME = 'train.csv'
DATA_VALID_FILE_NAME = 'valid.csv'
DATA_TEST_FILE_NAME = 'test.csv'
UNK_TOKEN = '<unk>'

TRAIN = True
VALID = True
TEST = True
VALID_DATA_SOURCE_TYPE = 2  # 0: valid file, 1: split from train data, 2: use test data, 3: split from test data
VALID_RATIO = 0.2

PRE_TRAIN_MODEL_BASE_PATH = '/home/ubuntu/likun/nlp_vectors'
PRE_TRAIN_MODEL_DIR = 'glove'
PRE_TRAIN_MODEL_NAME = 'glove.6B.200d.txt'

MODEL_SAVE_BASE_PATH = '/home/ubuntu/likun/nlp-practice/text_classify'
MODEL_NAME = 'zhnews-rnn.pt'

# Model Parameters
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 128
NUM_LAYERS = 2
EPOCH_SIZE = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

# build preprocess tokenizer
remove_strs = ['<br />', '(', ')', '"']
nlp = spacy.load('en')
# def tokenizer(text):
#     text = utils.remove_str_from_sentence(text, remove_strs)
#     return [token.text for token in nlp.tokenizer(text)]
def tokenizer(text):
    # text = utils.remove_str_from_sentence(text, remove_strs)
    return text.split()
user_stop_words = {'.', ','}
STOP_WORDS.update(user_stop_words)
stop_words = STOP_WORDS


# Pretrain Model
USE_PRE_TRAIN_MODEL = False
cache = '.vector_cache'
vector_path = os.path.join(PRE_TRAIN_MODEL_BASE_PATH, PRE_TRAIN_MODEL_DIR, PRE_TRAIN_MODEL_NAME)
vectors = Vectors(name=vector_path, cache=cache) if USE_PRE_TRAIN_MODEL else None

# Build Dataset
TEXT = data.Field(unk_token=UNK_TOKEN, tokenize=tokenizer, lower=False, stop_words=stop_words, batch_first=True)
LABEL = data.LabelField()
train_data = data.TabularDataset(path=os.path.join(DATA_BASE_PATH, DATA_DIR, DATA_TRAIN_FILE_NAME),
                                 format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
test_data = data.TabularDataset(path=os.path.join(DATA_BASE_PATH, DATA_DIR, DATA_TEST_FILE_NAME),
                                format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
if VALID_DATA_SOURCE_TYPE == 0:
    valid_data = data.TabularDataset(path=os.path.join(DATA_BASE_PATH, DATA_DIR, DATA_VALID_FILE_NAME), format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
elif VALID_DATA_SOURCE_TYPE == 1:
    train_data, valid_data = train_data.split(split_ratio=(1 - VALID_RATIO))
elif VALID_DATA_SOURCE_TYPE == 2:
    valid_data = test_data
elif VALID_DATA_SOURCE_TYPE == 3:
    test_data, valid_data = test_data.split(split_ratio=(1 - VALID_RATIO))
else:
    valid_data = None

TEXT.build_vocab(train_data, vectors=vectors)
LABEL.build_vocab(train_data)

print('Train Example: {}'.format('\n'.join(['{} ---- {}'.format(example.text, example.label) for example in train_data.examples[:5]])))
print('Valid Example: {}'.format('\n'.join(['{} ---- {}'.format(example.text, example.label) for example in valid_data.examples[:5]])))
print('Test Example: {}'.format('\n'.join(['{} ---- {}'.format(example.text, example.label) for example in test_data.examples[:5]])))

train_iter = data.BucketIterator(dataset=train_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text))
valid_iter = data.BucketIterator(dataset=valid_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text))
test_iter = data.Iterator(dataset=test_data, batch_size=BATCH_SIZE, sort=False)


# build model
from model import RNN, WordAVGModel, TextCNN
embedding_size = TEXT.vocab.vectors.shape[1] if USE_PRE_TRAIN_MODEL else EMBEDDING_SIZE
# model = RNN(input_size=len(TEXT.vocab), embedding_size=embedding_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=len(LABEL.vocab))
model = TextCNN(input_size=len(TEXT.vocab), embedding_size=embedding_size, output_size=len(LABEL.vocab), pooling_method='avg')
# model = WordAVGModel(vocab_size=len(TEXT.vocab), embedding_dim=embedding_size, output_dim=len(LABEL.vocab))
utils.weight_init(model)
if USE_PRE_TRAIN_MODEL:
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
if TRAIN:
    for epoch in range(1, 1 + EPOCH_SIZE):
        train_loss = []
        valid_loss = []
        valid_acc = 0
        model.train()
        for batch in tqdm(train_iter):
            model.zero_grad()
            text = batch.text.to(device)
            label = batch.label.to(device)

            output = model(text)

            loss = loss_function(output, label)
            train_loss.append(loss.data.tolist())
            loss.backward()
            optimizer.step()
        if VALID:
            model.eval()
            predict = []
            total = []
            for batch in tqdm(valid_iter):
                model.zero_grad()
                text = batch.text.to(device)
                label = batch.label.to(device)
                output = model(text)
                loss = loss_function(output, label)
                valid_loss.append(loss.data.tolist())
                total.extend(label.tolist())
                predict.extend(torch.argmax(output, dim=1).tolist())
            valid_acc, _, _, _ = utils.evaluate(total, predict)
        # scheduler.step()
        print("Epoch: {}, Train Loss: {}, Valid Loss: {}, Valid acc: {}".format(epoch, np.mean(train_loss), np.mean(valid_loss), valid_acc))
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_BASE_PATH, 'save_models', MODEL_NAME))
    print("Model saved in {}".format(os.path.join(MODEL_SAVE_BASE_PATH, 'save_models', MODEL_NAME)))
if TEST:
    if not TRAIN:
        model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_BASE_PATH, 'save_models', MODEL_NAME)))
    model.eval()
    total = []
    predict = []
    for batch in tqdm(test_iter):
        text = batch.text.to(device)
        label = batch.label.to(device)
        output = model(text)
        _, predict_label = torch.max(output, dim=1)
        total.extend(label.tolist())
        predict.extend(predict_label.tolist())
    res = utils.evaluate_report(total, predict)
    print(res)