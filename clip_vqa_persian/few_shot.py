import argparse
from ast import arg
import torch
import sys
import json
import pickle
from clipfamodel import get_model
from clipfadataset import create_test_generator, create_train_dataset
from clipfatrain import train
from clipfatest import test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='./dataset.json')
    parser.add_argument('--test_data', default='./dataset.json')
    parser.add_argument('--image_path', default='./images/')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='clip-persian_few_shot', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-5)
    # parser.add_argument('--save_every', type=int, default=200)
    # parser.add_argument('--prefix_length', type=int, default=10)
    # parser.add_argument('--prefix_length_clip', type=int, default=10)
    # parser.add_argument('--prefix_size', type=int, default=640)
    parser.add_argument('--bs', type=int, default=40)
    # parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    # parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    # parser.add_argument('--num_layers', type=int, default=8)
    # parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    # parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--model_weights_vision', default='SajjadAyoubi/clip-fa-vision')
    parser.add_argument('--model_weights_text', default='SajjadAyoubi/clip-fa-text')
    # parser.add_argument('--language', default="english", choices=('english', 'persian'))
    parser.add_argument('--shots', type=int, default=10)
    args = parser.parse_args()
    

    model = get_model(args.model_weights_vision, args.model_weights_text)
    _ , _ , training_set = create_train_dataset(args.train_data, args.image_path, args.shots)
    model = train(model, training_set, args.epochs, args.prefix, args.lr, args.bs)
    classes, label_to_index, test_generator = create_test_generator(args.test_data, args.image_path, args.bs)
    test(model, test_generator, classes, label_to_index)


if __name__ == '__main__':
    main()
