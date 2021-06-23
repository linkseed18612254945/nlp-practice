from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

def encode_fn(text_list, tokenizer, use_tqdm=True):
    all_input_ids = []
    iter_list = tqdm.tqdm(text_list) if use_tqdm else text_list
    for text in iter_list:
        input_ids = tokenizer.encode(
                        text,
                        truncation=True,
                        add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                        max_length=50,           # 设定最大文本长度
                        pad_to_max_length=True,   # pad到最大的长度
                        return_tensors='pt'       # 返回的类型为pytorch tensor
                   )
        all_input_ids.append(input_ids)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids


def predict(model, tokenizer, text, predict_batch_size=32):
    model.eval()
    if isinstance(text, str):
        text = [text]
    predict_ids = encode_fn(text, tokenizer, use_tqdm=False)
    predict_dataset = TensorDataset(predict_ids, torch.tensor([0] * len(text)))
    predict_dataloader = DataLoader(predict_dataset, batch_size=predict_batch_size, shuffle=False)
    predict_res_tags = []
    for i, batch in tqdm.tqdm(list(enumerate(predict_dataloader)), desc='Predicting'):
        with torch.no_grad():
            logits, feature = model(batch[0].to(model.device))
            logits = logits.detach().cpu().numpy()
            logits_ids = np.argmax(-logits, axis=2)
            predict_tokens = tokenizer
    return predict_res_tags

if __name__ == '__main__':

    path = '/home/ubuntu/likun/MT/opus-mt-en-zh'

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)

    test_sentence = ['Republicans in Washington are going to have a very hard time processing this. But the future is clear: we must be a working class party, not a Wall Street party']
    input_ids = encode_fn(test_sentence, tokenizer)