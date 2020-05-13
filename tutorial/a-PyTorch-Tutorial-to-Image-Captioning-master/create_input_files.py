from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='flickr8k',
                       karpathy_json_path='/home/ubuntu/likun/image_data/caption/dataset_flickr8k.json',
                       image_folder='/home/ubuntu/likun/image_data/flickr8k-images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/home/ubuntu/likun/image_data',
                       max_len=50)
