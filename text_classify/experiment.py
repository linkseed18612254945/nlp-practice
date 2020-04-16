import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torchtext import data, datasets
import spacy
from tqdm import tqdm
from spacy.symbols import ORTH

DATA_BASE_PATH = '/root/nlp_data'
# DATA_BASE_PATH = r'C:\Users\51694\Documents\nlp_data'
DATA_DIR = 'text_classify/aclImdb'
DATA_TRAIN_FILE_NAME = 'train.csv'
DATA_TEST_FILE_NAME = 'test.csv'

UNK_TOKEN = '<unk>'
BATCH_SIZE = 32

train = True
MODEL_PATH = 'save_models'
MODEL_NAME = 'imbd-rnn.pt'

# build tokenizer
nlp = spacy.load('en')
nlp.tokenizer.add_special_case(UNK_TOKEN, [{ORTH: UNK_TOKEN}])
nlp.tokenizer.add_special_case('<br/>', [{ORTH: '<br/>'}])
def tokenizer(text):
    return [token.text for token in nlp.tokenizer(text)]

# build dataset
TEXT = data.Field(unk_token=UNK_TOKEN, tokenize=tokenizer, batch_first=True)
LABEL = data.LabelField()
train_data = data.TabularDataset(path=os.path.join(DATA_BASE_PATH, DATA_DIR, DATA_TRAIN_FILE_NAME),
                                 format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
test_data = data.TabularDataset(path=os.path.join(DATA_BASE_PATH, DATA_DIR, DATA_TEST_FILE_NAME),
                                 format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
TEXT.build_vocab(train_data)
LABEL.build_vocab(train_data)
train_iter = data.BucketIterator(dataset=train_data, batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text))
test_iter = data.Iterator(dataset=test_data, batch_size=BATCH_SIZE, sort=False)

# build model
from text_classify.model import RNN, DNN
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
EPOCH_SIZE = 2
LEARNING_RATE = 1e-2
model = DNN(input_size=len(TEXT.vocab), embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS, output_size=len(LABEL.vocab))
# model = RNN(input_size=len(TEXT.vocab), embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
#             output_size=len(LABEL.vocab))
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

if train:
    for epoch in range(1, 1 + EPOCH_SIZE):
        train_loss = []
        for batch in tqdm(train_iter):
            optimizer.zero_grad()
            text = batch.text
            label = batch.label
            output = model(text)
            loss = loss_function(output, label)
            train_loss.append(loss.data.tolist())
            loss.backward()
            optimizer.step()
        print("Epoch: {}, Train Loss: {}".format(epoch, np.mean(train_loss)))
    torch.save(model.state_dict(), os.path.join(MODEL_PATH, MODEL_NAME))
    print("Model saved in {}".format(os.path.join(MODEL_PATH, MODEL_NAME)))