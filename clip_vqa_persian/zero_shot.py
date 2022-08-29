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
    parser.add_argument('--test_data', default='./dataset.json')
    parser.add_argument('--image_path', default='./images/')
    parser.add_argument('--bs', type=int, default=40)
    parser.add_argument('--model_weights_vision', default='SajjadAyoubi/clip-fa-vision')
    parser.add_argument('--model_weights_text', default='SajjadAyoubi/clip-fa-text')
    args = parser.parse_args()
    

    model = get_model(args.model_weights_vision, args.model_weights_text)
    classes, label_to_index, test_generator = create_test_generator(args.test_data, args.image_path, args.bs)
    test(model, test_generator, classes, label_to_index)


if __name__ == '__main__':
    main()
