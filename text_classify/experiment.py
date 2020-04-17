import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torchtext import data, datasets
import spacy
from tqdm import tqdm
from spacy.symbols import ORTH
import utils

USE_GPU = True
GPU_INDEX = 0
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(GPU_INDEX))
else:
    device = torch.device('cpu')

DATA_BASE_PATH = '/home/ubuntu/likun/nlp_data'
# DATA_BASE_PATH = r'C:\Users\51694\Documents\nlp_data'
DATA_DIR = 'text_classify/aclImdb'
DATA_TRAIN_FILE_NAME = 'train.csv'
DATA_TEST_FILE_NAME = 'test.csv'
TRAIN_VALID_SPLIT_RATIO = 0.8

UNK_TOKEN = '<unk>'
BATCH_SIZE = 32

MODEL_SAVE_BASE_PATH = '/home/ubuntu/likun/nlp-practice/text_classify'
MODEL_NAME = 'imbd-rnn.pt'
train = True
valid = True
test = True


# build tokenizer
nlp = spacy.load('en')
nlp.tokenizer.add_special_case(UNK_TOKEN, [{ORTH: UNK_TOKEN}])
nlp.tokenizer.add_special_case('<br/>', [{ORTH: '<br/>'}])
# def tokenizer(text):
#     return [token.text for token in nlp.tokenizer(text)]
def tokenizer(text):
    return text.split()

# build dataset
TEXT = data.Field(unk_token=UNK_TOKEN, tokenize=tokenizer, batch_first=True)
LABEL = data.LabelField()
train_data = data.TabularDataset(path=os.path.join(DATA_BASE_PATH, DATA_DIR, DATA_TRAIN_FILE_NAME),
                                 format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
test_data = data.TabularDataset(path=os.path.join(DATA_BASE_PATH, DATA_DIR, DATA_TEST_FILE_NAME),
                                 format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
if valid:
    train_data, valid_data = train_data.split(split_ratio=TRAIN_VALID_SPLIT_RATIO)
TEXT.build_vocab(train_data)
LABEL.build_vocab(train_data)
train_iter = data.BucketIterator(dataset=train_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text))
valid_iter = data.BucketIterator(dataset=valid_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text))
test_iter = data.Iterator(dataset=test_data, batch_size=BATCH_SIZE, sort=False)

# build model
from model import RNN
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
EPOCH_SIZE = 2
LEARNING_RATE = 1e-2
model = RNN(input_size=len(TEXT.vocab), embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
            output_size=len(LABEL.vocab))
utils.weight_init(model)
model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

if train:
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
        if valid:
            model.eval()
            predict = []
            total = []
            for batch in tqdm(valid_data):
                text = batch.text.to(device)
                label = batch.label.to(device)
                output = model(text)
                loss = loss_function(output, label)
                valid_loss.append(loss)
                total.extend(label.tolist())
                predict.extend(torch.argmax(output).tolist())
            valid_acc = utils.evaluate(total, predict)
        print("Epoch: {}, Train Loss: {}, Valid Loss: {}, Valid acc: {}".format(epoch, np.mean(train_loss), np.mean(valid_loss), valid_acc))
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_BASE_PATH, 'save_models', MODEL_NAME))
    print("Model saved in {}".format(os.path.join(MODEL_SAVE_BASE_PATH, 'save_models', MODEL_NAME)))
if test:
    if not train:
        model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_BASE_PATH, 'save_models', MODEL_NAME)))
    model.eval()
    total = []
    predict = []
    for batch in tqdm(train_iter):
        text = batch.text.to(device)
        label = batch.label.to(device)
        output = model(text)
        _, predict_label = torch.max(output, dim=1)
        total.extend(label.tolist())
        predict.extend(predict_label.tolist())
    res = utils.evaluate_report(total, predict)
    print(res)