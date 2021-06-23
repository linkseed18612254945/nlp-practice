import random
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import tqdm
import simple_utils
import os
import mlflow


def encode_fn(text_list, tokenizer, use_tqdm=True):
    all_input_ids = []
    iter_list = tqdm.tqdm(text_list) if use_tqdm else text_list
    for text in iter_list:
        input_ids = tokenizer.encode(
                        text,
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
    df_train = df_train.dropna(subset=[train_text_col])
    df_train = df_train.drop_duplicates(subset=[train_text_col])

    if balance:
        balance_df = pd.DataFrame()
        min_label_num = min(df_train[train_label_col].value_counts())
        for label in df_train[train_label_col].value_counts().keys():
            balance_df = balance_df.append(df_train[df_train[train_label_col] == label].sample(n=min_label_num))
        df_train = balance_df

        # harm_df = df_train[df_train[train_label_col] == 1]
        # no_harm_df = df_train[df_train[train_label_col] == 0]
        # if no_harm_df.shape[0] >= harm_df.shape[0]:
        #     no_harm_df = no_harm_df.sample(harm_df.shape[0])
        # else:
        #     harm_df = harm_df.sample(no_harm_df.shape[0])
        # df_train = harm_df.append(no_harm_df).sample(frac=1)

    if train_sample_num is not None:
        df_train = df_train.sample(n=train_sample_num)
    if train_label_col is None or train_label_col not in df_train.columns:
        df_train[train_label_col] = label_map[0]
    if label_map is not None:
        df_train[train_label_col] = df_train[train_label_col].apply(lambda x: label_map.index(x))
    train_input_ids = encode_fn(df_train[train_text_col], tokenizer)
    train_label = torch.tensor(df_train[train_label_col].tolist())
    train_dataset = TensorDataset(train_input_ids, train_label)

    df_test = pd.read_csv(test_path)
    df_test = df_test.dropna(subset=[test_text_col])
    df_test = df_test.drop_duplicates(subset=[test_text_col])

    if test_sample_num is not None:
        df_test = df_test.sample(n=test_sample_num)
    # test_input_ids = encode_fn(df_test[test_text_col], tokenizer)
    #
    # if test_label_col is not None and len(test_label_col) > 0:
    #     if label_map is not None:
    #         df_test[test_label_col] = df_test[test_label_col].apply(lambda x: label_map.index(x))
    #     test_label = torch.tensor(df_test[test_label_col].tolist())
    # else:
    #     test_label = torch.tensor([0] * df_test.shape[0])
    # test_dataset = TensorDataset(test_input_ids, test_label)

    if valid_percent > 0:
        train_size = int((1 - valid_percent) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_dataloader = None
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # if val_path is not None:
    #     df_val = pd.read_csv(val_path)
    #     df_val = df_val.dropna(subset=[train_text_col])
    #     df_val = df_val.drop_duplicates(subset=[train_text_col])
    #     if label_map is not None:
    #         df_val[train_label_col] = df_val[train_label_col].apply(lambda x: label_map.index(x))
    #     val_input_ids = encode_fn(df_val[train_text_col], tokenizer)
    #     val_label = torch.tensor(df_val[train_label_col].tolist())
    #     val_dataset = TensorDataset(val_input_ids, val_label)
    #     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # else:
    #     val_dataloader = None

    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, df_test

def train(train_data, model, valid_data=None):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    total_steps = len(train_data) * epoch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    valid_data = train_data if valid_data is None and valid_percent == 0 else valid_data
    for epoch in range(epoch_size):
        model.train()
        total_loss, total_val_loss = 0, 0
        for step, batch in tqdm.tqdm(list(enumerate(train_data))):
            model.zero_grad()
            attention_mask = (batch[0] > 0)
            loss, logits = model(batch[0].to(device), token_type_ids=None, attention_mask=attention_mask.to(device),
                                 labels=batch[1].to(device))
            total_loss += loss.item()
            loss.backward()
            torch.nn.simple_utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        predict = []
        true = []
        for i, batch in tqdm.tqdm(list(enumerate(valid_data))):
            with torch.no_grad():
                loss, logits = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                                     labels=batch[1].to(device))
                logits = np.argmax(logits.detach().cpu().numpy(), axis=1).tolist()
                label_ids = batch[1].to('cpu').numpy().tolist()
                predict.extend(logits)
                true.extend(label_ids)

        avg_train_loss = total_loss / len(train_data)
        avg_val_loss = total_val_loss / len(valid_data)
        if num_labels == 2:
            tp, fp, fn, tn, acc, p, r, f1 = simple_utils.matrix_evaluate(true, predict)
            print(f'Train loss     : {avg_train_loss}')
            print(f'Validation loss: {avg_val_loss}')
            res_str = f"Total: {len(predict)}, HarmTotal: {tp + fn}, HarmPercent: {(tp + fn) / len(predict):.3f} " \
                      f"-- TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn} -- " \
                      f"ACC: {acc:.3f} P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}"
            print(res_str)
            print('\n')
        else:
            report = simple_utils.evaluate_report(true, predict, list(range(num_labels)), label_map)
            print(report)

@simple_utils.logging_time_wrapper
def test(model, test_data, reverse=False):
    model.eval()
    predict_label = []
    true_label = []
    for i, batch in tqdm.tqdm(list(enumerate(test_data))):
        with torch.no_grad():
            _, logits = model(batch[0].to(model.device), token_type_ids=None,
                              attention_mask=(batch[0] > 0).to(model.device),
                              labels=batch[1].to(model.device))
            logits = np.argmax(logits.detach().cpu().numpy(), axis=1).tolist()

            label_ids = batch[1].to('cpu').numpy().tolist()
            predict_label.extend(logits)
            true_label.extend(label_ids)
    if num_labels == 2:
        if reverse:
            true_label = [1 - x for x in true_label]
            predict_label = [1 - x for x in predict_label]
        tp, fp, fn, tn, acc, p, r, f1 = simple_utils.matrix_evaluate(true_label, predict_label)
        res_str = f"Total: {len(predict_label)}, HarmTotal: {tp + fn}, HarmPercent: {(tp + fn) / len(predict_label):.3f} " \
                  f"-- TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn} -- " \
                  f"ACC: {acc:.3f} P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}"
        print(res_str)
        print('\n')
    else:
        report = simple_utils.evaluate_report(true_label, predict_label, target_names=label_map)
        print(report)
    return true_label, predict_label

@simple_utils.logging_time_wrapper
def predict(model, tokenizer, text, predict_batch_size=32):
    model.eval()
    if isinstance(text, str):
        text = [text]
    predict_ids = encode_fn(text, tokenizer, use_tqdm=False)
    predict_dataset = TensorDataset(predict_ids, torch.tensor([0] * len(text)))
    predict_dataloader = DataLoader(predict_dataset, batch_size=predict_batch_size, shuffle=False)
    predict_res_tags = []
    predict_res_scores = []
    for i, batch in tqdm.tqdm(list(enumerate(predict_dataloader)), desc='Predicting'):

        with torch.no_grad():
            _, logits = model.forward(batch[0].to(model.device), token_type_ids=None,
                              attention_mask=(batch[0] > 0).to(model.device),
                              labels=batch[1].to(model.device))
            logits = logits.detach().cpu().numpy()

            logits_labels = np.argsort(-logits, axis=1)
            sorted_scores = np.exp(-np.sort(-logits))
            sum_scores = np.repeat(np.sum(sorted_scores, axis=1), sorted_scores.shape[1]).reshape(sorted_scores.shape)
            sorted_scores = sorted_scores / sum_scores
            predict_res_scores.append(sorted_scores)
            predict_res_tags.append(logits_labels)
    predict_res_tags = np.vstack(predict_res_tags)
    predict_res_scores = np.vstack(predict_res_scores)
    return predict_res_tags, predict_res_scores


def multi_label_test(true_tags, predict_tags):
    report_true_tags = []
    report_predict_tags = []
    for true_tag, predict_tag in zip(true_tags, predict_tags):
        same_labels = set(true_tag) & set(predict_tag)
        if len(same_labels) > 0:
            same_label = same_labels.pop()
            report_true_tags.append(same_label)
            report_predict_tags.append(same_label)
        else:
            report_true_tags.append(true_tag[0])
            report_predict_tags.append(predict_tag[0])
    return report_true_tags, report_predict_tags



if __name__ == '__main__':
    # label_map = ['社会', '经济金融', '互联网', '公共安全', '教育', '文化艺术', '国际政治',
    #              '法律司法', '国内政治', '科学技术', '行业产业', '环境能源', '医药卫生', '农业农村',
    #              '休闲娱乐', '军事', '民族宗教', '体育']
    # label_map = ['涉港', '中美关系', '涉台', '政治攻击', '政治局委员相关', '涉新冠疫情', '个人攻击',
    #              '民族宗教、人权民主', '外交关系', '热点事件', '涉六四', '重点人', '领导人', '民生经济', '中印关系',
    #              '意识形态', '重要会议、重要活动', '法轮功', '华为相关', '维权相关', '高层权斗', '一带一路', '文革相关']
    with open('/home/ubuntu/likun/nlp_data/text_classify/work_data/finance/warning_factors.txt', 'r', encoding='utf-8') as f:
        label_map = f.read().splitlines()
    train_model = False
    balance = False
    train_sample_num = 10
    test_sample_num = None
    valid_percent = 0.1
    train_text_col = 'token'
    train_label_col = 'label'
    test_text_col = 'token'
    test_label_col = None
    num_labels = len(label_map)
    train_path = '/home/ubuntu/likun/nlp_data/text_classify/work_data/finance/yq-2021-train-category.csv'
    val_path = None
    test_path = '/home/ubuntu/likun/nlp_data/text_classify/work_data/finance/wx_guomei.csv'
    res_path = '/home/ubuntu/likun/nlp_predict_res/wx_guomei_res.csv'

    # model_name = 'bert-google-finetune-yqzx-18category-all-5epoch-0923.pt'
    # model_name = 'bert-google-finetune-finance-emotion-title-0412.pt'
    model_name = 'bert-wwm-finance-newdata0401-category-token-0512.pt'

    seed = 49
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    USE_GPU = True
    GPU_INDEX = 0
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(GPU_INDEX))
    else:
        device = torch.device('cpu')

    epoch_size = 6
    batch_size = 32

    model_save_path = '/home/ubuntu/likun/nlp_save_kernels'
    bert_model_path = '/home/ubuntu/likun/nlp_pretrained/bert-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(bert_model_path, num_labels=num_labels,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    model.to(device)
    if train_model:
        train_dataloader, val_dataloader, df_test = cn_data_build(tokenizer)
        train(train_dataloader, model, val_dataloader)
        torch.save(model.state_dict(), os.path.join(model_save_path, model_name))
    else:
        df_test = pd.read_csv(test_path)
        df_test = df_test.dropna(subset=[test_text_col])
        df_test = df_test.drop_duplicates(subset=[test_text_col])
        if test_sample_num is not None:
            df_test = df_test.sample(n=test_sample_num)
        # save_point = torch.load(os.path.join(model_save_path, model_name))
        save_point = torch.load('/home/ubuntu/likun/nlp_save_kernels/bert-wwm-finance-newdata0401-token-0512.pt')
        model.load_state_dict(save_point, strict=False)

    predict_tag_num = 2
    predict_tags, predict_probs = predict(model, tokenizer, df_test[test_text_col].tolist())

    good_tags = predict_tags[:, :predict_tag_num]
    good_probs = predict_probs[:, :predict_tag_num]

    good_tags = [';'.join(map(lambda x: str(label_map[x]), c)) for c in good_tags.tolist()]
    good_probs = [';'.join(map(lambda x: "{:.3f}".format(x), c)) for c in good_probs.tolist()]
    df_test['predict_tags'] = good_tags
    df_test['predict_probs'] = good_probs
    if test_label_col is not None:
        true_labels = [label_map[label] if isinstance(label, int) else label for label in df_test[test_label_col]]
        # best_predict_tags = [label_map[c[0]] for c in predict_tags[:, :1].tolist()]

        true_labels = [true_label.split(',') for true_label in true_labels]
        best_predict_tags = [[label_map[c[i]] for i in range(predict_tag_num)] for c in predict_tags[:, :predict_tag_num].tolist()]

        true_labels, best_predict_tags = multi_label_test(true_labels, best_predict_tags)

        report = simple_utils.evaluate_report(true_labels, best_predict_tags, label_map)
        # df_test[test_label_col] = df_test[test_label_col].apply(lambda x: label_map[x])
        print(report)
    df_test.to_csv(res_path, index=False)
    # test_df[['ID', 'TITLE', 'PUBLISH_TIME', 'CONTEXT_TEXT', 'FULLNAME', 'WARNINGFACTORS', 'predict_tags_1', 'predict_prob_2']].to_csv(res_path, index=False)[['ID', 'TITLE', 'PUBLISH_TIME', 'CONTEXT_TEXT', 'FULLNAME', 'WARNINGFACTORS', 'predict_tags_1', 'predict_prob_2']].to_csv(res_path, index=False)


