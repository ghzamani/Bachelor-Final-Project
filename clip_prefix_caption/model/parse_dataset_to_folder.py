import torch
import skimage.io as io
import clip
# from transformers import CLIPModel, CLIPFeatureExtractor
from transformers import AutoModel, CLIPVisionModel, RobertaModel, AutoTokenizer, CLIPFeatureExtractor
from transformers import AutoTokenizer, GPT2Tokenizer
from torch import nn
from transformers import CLIPConfig, CLIPModel
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import numpy as np


def get_persian_model(device):
    # config = {'num_hidden_layers': 0,
    #           'max_position_embeddings': 0,
    #           'vocab_size': 0,
    #           'hidden_size': 1,
    #           'patch_size': 1,
    #           }
    # DUMMY_CONFIG = CLIPConfig(text_config_dict=config,
    #                           vision_config_dict=config)
    # clip = CLIPModel(config=DUMMY_CONFIG)
    # # convert projectors to Identity
    # clip.text_projection = nn.Identity()
    # clip.visual_projection = nn.Identity()
    vision_encoder = CLIPVisionModel.from_pretrained('SajjadAyoubi/clip-fa-vision').to(device)
    preprocess = CLIPFeatureExtractor.from_pretrained('SajjadAyoubi/clip-fa-vision')
    # text_encoder = RobertaModel.from_pretrained('SajjadAyoubi/clip-fa-text')
    # # tokenizer = AutoTokenizer.from_pretrained('SajjadAyoubi/clip-fa-text')
    # assert text_encoder.config.hidden_size == vision_encoder.config.hidden_size

    # clip.text_model = text_encoder
    # clip.vision_model = vision_encoder

    # preprocess = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
    return vision_encoder, preprocess

def main(clip_model_type: str, language: str, dataset_json_path, image_path, out_path):
    # device = torch.device('cuda:0')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device", device)
    clip_model_name = clip_model_type.replace('/', '_')
    # out_path = f"{out_path}{clip_model_name}_train.pkl"
    
    if language == "english":
        clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    else:
        clip_model, preprocess = get_persian_model(device)
        tokenizer = AutoTokenizer.from_pretrained('bolbolzaban/gpt2-persian')
        # preprocess = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        # clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # clip_model, preprocess = clipfa.load(clip_model_type, device=device, jit=False)
    with open(dataset_json_path, 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    max_seq_len = 0
    all_caption_tokens_len = []
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image"]
        filename = f"{image_path}{img_id}"
        # if not os.path.isfile(filename):
        #     filename = f"./data/dataset/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        image = io.imread(filename)
        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)
        if language == "english":
            image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
            with torch.no_grad():
                prefix = clip_model.encode_image(image).cpu()
        else:
            inputs = preprocess(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = clip_model(**inputs)
                prefix = outputs.pooler_output.cpu()
        d["clip_embedding"] = prefix
        d["captions_tokens"] = torch.tensor(tokenizer.encode(d['caption']), dtype=torch.int64)
        all_caption_tokens_len.append(len(d["captions_tokens"]))
        max_seq_len = max(max_seq_len, d["captions_tokens"].shape[0])
        # all_embeddings.append(prefix)
        # all_captions.append(d)
        # if (i + 1) % 10000 == 0:
        with open(f"{out_path}{i}.pkl", 'wb') as f:
            pickle.dump(d, f)

    all_len = torch.tensor(all_caption_tokens_len).float()
    if len(all_len) == 1:
        processed_max_seq_len = int(all_len.max())
    else:
        processed_max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
    with open(f"{out_path}dataset_info.pkl", 'wb') as f:
        pickle.dump({"processed_max_seq_len": processed_max_seq_len, "max_seq_len": max_seq_len, "dataset_length": len(data)}, f)

    print('Done')
    print("%0d embeddings saved " % len(data))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="RN50x4", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--language', default="english", choices=('english', 'persian'))
    parser.add_argument('--dataset_json', default='./data/dataset/dataset_train.json')
    parser.add_argument('--categories_path', default='')
    parser.add_argument('--image_path', default='./data/dataset/train/')
    parser.add_argument('--out_folder', default='./')
    args = parser.parse_args()
    
    if args.categories_path != '':
        print("Using categories path will ignore args dataset_json")
        categories = ['cars', 'ceremonies', 'food', 'indoor', 'ashkhas', 'sport']
        for category in categories:
            print("###### Using category", category, "#######")
            args.dataset_json = f'{args.categories_path}{category}_train'
            if args.language == 'english':
                args.dataset_json = args.dataset_json + "_eng"
            args.train_pickle = args.dataset_json + ".pkl"
            args.dataset_json = args.dataset_json + ".json"
            main(args.clip_model_type, args.language, args.dataset_json, args.image_path, args.train_pickle)
        
    else:
        main(args.clip_model_type, args.language, args.dataset_json, args.image_path, args.out_folder)

    # with open(f"ViT-B_32_train.pkl","rb") as f:
    #     p = pickle.load(f)
    #     print(p['clip_embedding'].shape)
