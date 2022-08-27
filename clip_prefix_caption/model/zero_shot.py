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
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--model_weights', default='')
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--language', default="english", choices=('english', 'persian'))
    args = parser.parse_args()
    prefix_length = args.prefix_length
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                num_layers=args.num_layers, mapping_type=args.mapping_type, is_eng= (args.language == "english"))
    if args.model_weights != "":
        print("Using pretrained weights in file: ", args.model_weights)
        model.load_state_dict(torch.load(args.model_weights, map_location= torch.device("cpu")))

    predictor = Predictor(args.model_weights, mapping_type=args.mapping_type,clip_length=args.prefix_length_clip, num_layers=args.num_layers, is_eng= (args.language == "english"), training_model=model)
    predictions, targets = predictor.test(args.test_data, use_beam_search=True)
    print(evaluate_metrics(predictions, targets, is_eng=(args.language == "english")))


if __name__ == '__main__':
    main()
