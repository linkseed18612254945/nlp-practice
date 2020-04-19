import jieba
import pandas as pd
import tqdm
import os

DATA_BASE_PATH = '/home/ubuntu/likun/nlp_data'
DATA_DIR = 'text_classify/car_comments'
file_name = 'test.tsv'
target_name = 'test.csv'

file_path = os.path.join(DATA_BASE_PATH, DATA_DIR, file_name)
target_path = os.path.join(DATA_BASE_PATH, DATA_DIR, target_name)

def build_tokenize_csv(df, text_label='text'):
    tokens = []
    for sentence in tqdm.tqdm(df[text_label]):
        token = ' '.join(jieba.cut(sentence))
        tokens.append(token)
    df[text_label] = tokens
    return df


if __name__ == '__main__':
    df = pd.read_csv(file_path, delimiter='\t')
    df = build_tokenize_csv(df)
    df.to_csv(target_path, index=False)