import numpy as np
import torch
from torch import nn
from torch import optim
import os
from torchtext import data, datasets
import spacy
from spacy.symbols import ORTH
from tqdm import tqdm
import utils

USE_GPU = True
USE_GPU_INDEX = 0
# device choose
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(USE_GPU_INDEX))
else:
    device = torch.device('cpu')

# DATA_BASE_PATH = '/root/nlp_data'
DATA_BASE_PATH = '/home/ubuntu/likun/nlp_data'
DATA_DIR = 'language_model/ptb'
DATA_TRAIN_FILE_NAME = 'ptb.train.txt'
DATA_VALID_FILE_NAME = 'ptb.valid.txt'
DATA_TEST_FILE_NAME = 'ptb.test.txt'

# BASE PARAMETER
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
BATCH_SIZE = 32
BPTT_LEN = 30

train = True
valid = True

# build tokenizer
nlp = spacy.load('en')
nlp.tokenizer.add_special_case(UNK_TOKEN, [{ORTH: UNK_TOKEN}])
def tokenizer(text):
    return [token.text for token in nlp.tokenizer(text)]

# build DATASET
TEXT = data.Field(eos_token=EOS_TOKEN, init_token=BOS_TOKEN, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, batch_first=True)
train_data = datasets.LanguageModelingDataset(path=os.path.join(DATA_BASE_PATH, DATA_DIR, DATA_TRAIN_FILE_NAME), text_field=TEXT)
valid_data = datasets.LanguageModelingDataset(path=os.path.join(DATA_BASE_PATH, DATA_DIR, DATA_VALID_FILE_NAME), text_field=TEXT)
test_data = datasets.LanguageModelingDataset(path=os.path.join(DATA_BASE_PATH, DATA_DIR, DATA_TEST_FILE_NAME), text_field=TEXT)
TEXT.build_vocab(train_data)
train_iter = data.BPTTIterator(dataset=train_data, batch_size=BATCH_SIZE, bptt_len=BPTT_LEN)
valid_iter = data.BPTTIterator(dataset=valid_data, batch_size=BATCH_SIZE, bptt_len=BPTT_LEN)
test_iter = data.BPTTIterator(dataset=test_data, batch_size=BATCH_SIZE, bptt_len=BPTT_LEN)

# build model
MODEL_SAVE_BASE_PATH = '/home/ubuntu/likun/nlp-practice/language_model'
MODEL_NAME = "PTB-RNN-KERNEL.pt"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_BASE_PATH, 'save_models', MODEL_NAME)
from model import RNN
NUM_EPOCH = 10
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
LEARNING_RATE = 1e-2
model = RNN(vocab_size=len(TEXT.vocab), hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE, num_layers=NUM_LAYERS)
utils.weight_init(model)
model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# train model
if train:
    for epoch in range(1, NUM_EPOCH + 1):
        train_losses = []
        valid_losses = []

        model.train()
        for batch in tqdm(train_iter):
            optimizer.zero_grad()
            text = batch.text.to(device)
            target = batch.target.to(device)
            target = target.contiguous().view(-1)
            output = model(text)
            loss = loss_function(output, target)
            train_losses.append(loss.data.tolist())
            loss.backward()
            optimizer.step()

        if valid:
            model.eval()
            for batch in tqdm(valid_iter):
                optimizer.zero_grad()
                text = batch.text.to(device)
                target = batch.target.to(device)
                target = target.view(-1)
                output = model(text)
                loss = loss_function(output, target)
                valid_losses.append(loss.data.tolist())
        print("EPOCH: {}, TRAINING PERPLEXITY: {:.2f}, VALIDATION PERPLEXITY: {:.2f}"
              .format(epoch, np.exp(np.mean(train_losses)), np.exp(np.mean(valid_losses)) if valid else 'NONE'))
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved in {}".format(MODEL_SAVE_PATH))

# test model
test_perplexity = 0
if not train:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    test_losses = []
    for batch in tqdm(test_iter):
        text = batch.text
        target = batch.target.view(-1)
        output = model(text)
        loss = loss_function(output, target)
        test_losses.append(loss)
    test_perplexity = np.exp(np.mean(test_losses))
    print("TEST PERPLEXITY: {:.2f}".format(test_perplexity))





