from torch.utils.data import Dataset
import os
import json
import torch
from torchtext import data, datasets
from torchtext.vocab import Vectors
from random import seed, choice, sample
import h5py
from imageio import imread
from PIL import Image
import tqdm
import numpy as np

def create_image_data_file(dataset, data_folder_path, caption_path, output_folder, captions_per_image=3):
    seed(123)

    with open(caption_path, 'r', encoding='utf-8') as j:
        captions = json.load(j)

    train_image_paths = []
    train_captions = []
    val_image_paths = []
    val_captions = []
    test_image_paths = []
    test_captions = []
    print("Reading captions, get the image path")
    for image_info in tqdm.tqdm(captions['images']):
        sentences_info = image_info['sentences']
        sentences = '|||'.join([sentence_info['raw'] for sentence_info in sentences_info])
        filename = image_info['filename']
        split = image_info['split']
        image_path = os.path.join(data_folder_path, filename)
        if split in ('train', 'restval'):
            train_image_paths.append(image_path)
            train_captions.append(sentences)
        elif split == 'val':
            val_image_paths.append(image_path)
            val_captions.append(sentences)
        elif split == 'test':
            test_image_paths.append(image_path)
            test_captions.append(sentences)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for impaths, imcaps, split in [(train_image_paths, train_captions, 'TRAIN'),
                                   (val_image_paths, val_captions, 'VAL'),
                                   (test_image_paths, test_captions, 'TEST')]:
        assert len(impaths) == len(imcaps)
        # 存储对应caption文本数据
        caption_text_path = os.path.join(output_folder, "{}_CAPTIONS_{}.hdf5".format(dataset, split))
        with open(caption_text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(imcaps))

        # 创建图像hdf5数据文件
        image_hdf5_path = os.path.join(output_folder, "{}_IMAGES_{}.hdf5".format(dataset, split))
        with h5py.File(image_hdf5_path, 'a') as h:
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')
            print("\nReading %s images and captions, storing to file...\n" % split)

            for i, path in tqdm.tqdm(list(enumerate(impaths))):
                img = imread(path)

                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = np.array(Image.fromarray(img).resize((256, 256)))
                img = img.transpose((2, 0, 1))
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255
                images[i] = img

def tokenizer(text):
    # text = utils.remove_str_from_sentence(text, remove_strs)
    return text.split()


class ImageCaptionDataset(Dataset):
    def __init__(self, data_folder, dataset_name, split, text_vector=None, device=None):
        vectors = Vectors(name=text_vector, cache='.vector_cache') if text_vector is not None else None
        self.img_path = os.path.join(data_folder, "{}_IMAGES_{}.hdf5".format(dataset_name, split))
        self.caption_path = os.path.join(data_folder, "{}_CAPTIONS_{}.hdf5".format(dataset_name, split))
        self.h = h5py.File(self.img_path, 'r')
        self.imgs = self.h['images']
        with open(self.caption_path, 'r', encoding='utf-8') as f:
            self.captions = f.read().splitlines()
        assert len(self.captions) == len(self.imgs)
        self.caption2img = []
        self.caption_tokens = []
        for i, caption in enumerate(self.captions):
            sentences = caption.split('|||')
            for sentence in sentences:
                self.caption_tokens.append(tokenizer(sentence))
                self.caption2img.append(i)
        self.max_length = max([len(token) for token in self.caption_tokens])
        self.corpus = data.Field(eos_token='<eos>', init_token='<bos>', unk_token='<unk>', sequential=True, fix_length=self.max_length)
        self.corpus.build_vocab(self.caption_tokens, vectors=vectors)
        self.caption_tokens = self.corpus.pad(self.caption_tokens)

    def __getitem__(self, i):
        img = torch.FloatTensor(self.imgs[self.caption2img[i]] / 255)
        caption = self.corpus.numericalize([self.caption_tokens[i]]).squeeze()
        return img, caption

    def __len__(self):
        return len(self.caption_tokens)



if __name__ == '__main__':
    caption_path = '/home/ubuntu/likun/image_data/caption/dataset_flickr8k.json'
    data_folder_path = '/home/ubuntu/likun/image_data/flickr8k-images'
    output_path = '/home/ubuntu/likun/image_data/image_caption_attention'
    text_vector_path = '/home/ubuntu/likun/nlp_vectors/glove/glove.6B.100d.txt'

    # create_image_data_file('flickr8k', data_folder_path, caption_path, output_path)
    dataset_name = 'flickr8k'
    split = 'TRAIN'
    caption_dataset = ImageCaptionDataset(output_path, dataset_name, split, text_vector=None)

    train_loader = torch.utils.data.DataLoader(caption_dataset, batch_size=32, shuffle=True)
    x = caption_dataset.corpus.numericalize(arr=[caption_dataset.caption_tokens[10], caption_dataset.caption_tokens[1]])

