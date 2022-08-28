import argparse
from ast import arg
from train import ClipCocoDataset, ClipCaptionPrefix, ClipCaptionModel, MappingType, train
import torch
import sys
from predict import Predictor
from metrics import evaluate_metrics
import json
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', default='./data/coco/oscar_split_train.pkl')
    parser.add_argument('--image_path', default='./data/coco/')
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--prefix_size', type=int, default=640)
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--model_weights', default='')
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--language', default="english", choices=('english', 'persian'))
    parser.add_argument('--clip_model_type', default="RN50x4", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    # prefix_length = args.prefix_length
    # # prefix_dim = 640 if args.is_rn else 512
    # prefix_dim = args.prefix_size
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]

    predictor = Predictor(args.model_weights, mapping_type=args.mapping_type,clip_length=args.prefix_length_clip, num_layers=args.num_layers,
     is_eng= (args.language == "english"),prefix_length=args.prefix_length, prefix_size=args.prefix_size, clip_model=args.clip_model_type)
    predictions, targets = predictor.test(args.test_data, args.image_path, use_beam_search=True)
    print(evaluate_metrics(predictions, targets, is_eng=(args.language == "english")))


if __name__ == '__main__':
    main()
