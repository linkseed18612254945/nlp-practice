import random
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer
from transformers import BertForNextSentencePrediction, AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import tqdm
import utils
import os
import mlflow


def encode_fn(prompt_list, next_sentence_list, tokenizer, use_tqdm=True):
    all_input_ids = []
    iter_list = tqdm.tqdm(list(zip(prompt_list, next_sentence_list)) if use_tqdm else zip(prompt_list, next_sentence_list))
    for prompt, next_sentence in iter_list:
        input_ids = tokenizer.encode(
                        prompt,
                        next_sentence,
                        truncation=True,
                        add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                        max_length=128,           # 设定最大文本长度
                        pad_to_max_length=True,   # pad到最大的长度
                        return_tensors='pt'       # 返回的类型为pytorch tensor
                   )
        all_input_ids.append(input_ids)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids


def cn_data_build(tokenizer):
    df_train = pd.read_csv(train_path)

    if balance:
        balance_df = pd.DataFrame()
        min_label_num = min(df_train[label_col].value_counts())
        for label in df_train[label_col].value_counts().keys():
            balance_df = balance_df.append(df_train[df_train[label_col] == label].sample(n=min_label_num))
        df_train = balance_df

    if train_sample_num is not None:
        df_train = df_train.sample(n=train_sample_num)

    train_input_ids = encode_fn(df_train[prompt_col], df_train[next_sentence_col], tokenizer)
    train_label = torch.tensor(df_train[label_col].tolist())
    train_dataset = TensorDataset(train_input_ids, train_label)

    df_test = pd.read_csv(test_path)
    if test_sample_num is not None:
        df_test = df_test.sample(n=test_sample_num)
    test_input_ids = encode_fn(df_test[prompt_col], df_test[next_sentence_col], tokenizer)
    test_label = torch.tensor(df_test[label_col].tolist())
    test_dataset = TensorDataset(test_input_ids, test_label)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader, df_test

def train(train_data, model, valid_data=None):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    total_steps = len(train_data) * epoch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    valid_data = train_data if valid_data is None else valid_data
    for epoch in range(epoch_size):
        model.train()
        total_loss, total_val_loss = 0, 0
        for step, batch in tqdm.tqdm(list(enumerate(train_data))):
            model.zero_grad()
            attention_mask = (batch[0] > 0)
            loss, logits = model(batch[0].to(device), token_type_ids=None, attention_mask=attention_mask.to(device),
                                 next_sentence_label=batch[1].to(device))
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        predict = []
        true = []
        for i, batch in tqdm.tqdm(list(enumerate(valid_data))):
            with torch.no_grad():
                loss, logits = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                                     next_sentence_label=batch[1].to(device))
                logits = np.argmax(logits.detach().cpu().numpy(), axis=1).tolist()
                label_ids = batch[1].to('cpu').numpy().tolist()
                predict.extend(logits)
                true.extend(label_ids)

        avg_train_loss = total_loss / len(train_data)
        avg_val_loss = total_val_loss / len(valid_data)
        tp, fp, fn, tn, acc, p, r, f1 = utils.matrix_evaluate(true, predict)
        print(f'Train loss     : {avg_train_loss}')
        print(f'Validation loss: {avg_val_loss}')
        res_str = f"Total: {len(predict)}, HarmTotal: {tp + fn}, HarmPercent: {(tp + fn) / len(predict):.3f} " \
                  f"-- TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn} -- " \
                  f"ACC: {acc:.3f} P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}"
        print(res_str)
        print('\n')

@utils.logging_time_wrapper
def evaluate(model, test_data, reverse=False):
    model.eval()
    predict_label = []
    true_label = []
    for i, batch in tqdm.tqdm(list(enumerate(test_data))):
        with torch.no_grad():
            _, logits = model(batch[0].to(model.device), token_type_ids=None,
                              attention_mask=(batch[0] > 0).to(model.device), next_sentence_label=batch[1].to(device))
            logits = np.argmax(logits.detach().cpu().numpy(), axis=1).tolist()

            label_ids = batch[1].to('cpu').numpy().tolist()
            predict_label.extend(logits)
            true_label.extend(label_ids)
    if reverse:
        true_label = [1 - x for x in true_label]
        predict_label = [1 - x for x in predict_label]
    tp, fp, fn, tn, acc, p, r, f1 = utils.matrix_evaluate(true_label, predict_label)
    res_str = f"Total: {len(predict_label)}, HarmTotal: {tp + fn}, HarmPercent: {(tp + fn) / len(predict_label):.3f} " \
              f"-- TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn} -- " \
              f"ACC: {acc:.3f} P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}"
    print(res_str)
    print('\n')
    return true_label, predict_label

@utils.logging_time_wrapper
def predict(model, tokenizer, left_sentences, right_sentences, predict_batch_size=32):
    model.eval()
    if isinstance(left_sentences, str):
        left_sentences = [left_sentences] * len(right_sentences)
    predict_ids = encode_fn(left_sentences, right_sentences, tokenizer, use_tqdm=False)
    predict_dataset = TensorDataset(predict_ids, torch.tensor([0] * len(right_sentences)))
    predict_dataloader = DataLoader(predict_dataset, batch_size=predict_batch_size, shuffle=False)
    predict_res_scores = []
    for i, batch in tqdm.tqdm(list(enumerate(predict_dataloader)), desc='Predicting'):
        with torch.no_grad():
            _, logits = model(batch[0].to(model.device), token_type_ids=None, attention_mask=(batch[0] > 0).to(model.device),
                           next_sentence_label=batch[1].to(device))
            logits = logits.detach().cpu().numpy()
            predict_res_scores.append(logits[:, 1])
    predict_res_scores = np.vstack(predict_res_scores)
    return predict_res_scores

if __name__ == '__main__':
    train_model = False
    balance = False
    train_sample_num = 10
    test_sample_num = 5000
    prompt_col = 'text_left'
    next_sentence_col = 'text_right'
    label_col = 'label'
    train_path = '/home/ubuntu/likun/nlp_data/text_match/text_matching_kg_train.csv'
    test_path = '/home/ubuntu/likun/nlp_data/text_match/text_matching_kg_test.csv'
    res_path = '/home/ubuntu/likun/nlp_predict_res/matching.csv'

    model_name = 'text-matching-kg-5w-0531.pt'

    seed = 49
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    USE_GPU = True
    GPU_INDEX = 1
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(GPU_INDEX))
    else:
        device = torch.device('cpu')

    epoch_size = 2
    batch_size = 32

    model_save_path = '/home/ubuntu/likun/nlp_save_kernels'
    bert_model_path = '/home/ubuntu/likun/nlp_pretrained/bert-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=True)
    model = BertForNextSentencePrediction.from_pretrained(bert_model_path,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    model.to(device)
    if train_model:
        train_dataloader, test_dataloader, df_test = cn_data_build(tokenizer)
        train(train_dataloader, model, test_dataloader)
        torch.save(model.state_dict(), os.path.join(model_save_path, model_name))
    else:
        df_test = pd.read_csv("/home/ubuntu/likun/nlp_data/text_match/results-kgSmall.csv")
        if test_sample_num is not None:
            df_test = df_test.sample(n=test_sample_num).reset_index().drop('index', axis=1)
        save_point = torch.load('/home/ubuntu/likun/nlp_save_kernels/bert-wwm-finance-newdata0401-token-0512.pt')
        model.load_state_dict(save_point, strict=False)
    df_test['token'] = df_test['name_token'] + ' ' + df_test['description_token']
    left_sentence = df_test['token'][:1].to_list() * df_test.shape[0]
    right_sentences = df_test["token"].to_list()
    scores = predict(model, tokenizer, left_sentence, right_sentences)
    top10_index = np.argsort(-scores)[:10]
    top10_scores = scores[top10_index]
    top10_text = df_test["token"][top10_index]