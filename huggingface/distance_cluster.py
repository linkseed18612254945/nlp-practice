from transformers import BertTokenizer, BertModel
import pandas as pd
import tqdm
import torch
import numpy as np

USE_GPU = True
GPU_INDEX = 1
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(GPU_INDEX))
else:
    device = torch.device('cpu')

def get_vector(text, tokenizer, model):
    text = text[:510]
    input_ids = tokenizer(text, return_tensors="pt")
    vector = model.forward(**input_ids)[0]
    vector = torch.mean(vector, axis=1).squeeze().detach().numpy()
    return vector

def distance(center, sentence):
    x = sentence - center
    return np.linalg.norm(x, 2)


data_path = '/home/ubuntu/likun/nlp_data/cluster/li_event.csv'
df = pd.read_csv(data_path)
test_text_col = 'use'

kws_path = '/home/ubuntu/likun/nlp_data/cluster/event_sentiment_kws.csv'
kws = pd.read_csv(kws_path)
sentiments = ['正向', '中立', '负向']
centers = [kws[s][0].replace(';', ' ') for s in sentiments]


bert_model_path = '/home/ubuntu/likun/nlp_pretrained/bert-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=True)
model = BertModel.from_pretrained(bert_model_path)

sentence_vectors = [get_vector(text, tokenizer, model) for text in tqdm.tqdm(df[test_text_col].tolist())]
center_vectors = [get_vector(text, tokenizer, model) for text in tqdm.tqdm(centers)]

distances = []
positions = []
for sentence in sentence_vectors:
    center_dis = []
    for center in center_vectors:
        center_dis.append(distance(sentence, center))
    positions.append(sentiments[np.argmin(center_dis)])
    distances.append(center_dis)
df['类别'] = positions
df.to_csv('li_event_res.csv', index=False, encoding='utf-8')