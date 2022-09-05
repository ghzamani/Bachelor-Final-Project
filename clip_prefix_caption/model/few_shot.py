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
    parser.add_argument('--train_data', default='./data/coco/oscar_split_train.pkl')
    parser.add_argument('--test_data', default='./data/coco/oscar_split_train.pkl')
    parser.add_argument('--image_path', default='./data/coco/')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--prefix', default='few_shot', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--prefix_size', type=int, default=640)
    parser.add_argument('--bs', type=int, default=40)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--model_weights', default='')
    parser.add_argument('--categories_path', default='')
    parser.add_argument('--language', default="english", choices=('english', 'persian'))
    parser.add_argument('--shots', type=int, default=10)
    parser.add_argument('--clip_model_type', default="RN50x4", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    prefix_length = args.prefix_length
    prefix_dim = args.prefix_size
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]

    if args.categories_path != '':
        print("Using categories path will ignore args test_data and train_data")
        categories = ['cars', 'ceremonies', 'food', 'indoor', 'ashkhas', 'sport']
        args.prefix = f'_{args.shots}_{args.language}'
        metrics_dict = {}
        for category in categories:
            print("###### Using category", category, "#######")
            args.test_data = f'{args.categories_path}{category}_test'
            args.train_data = f'{args.categories_path}{category}_train'
            if args.language == 'english':
                args.test_data = args.test_data + "_eng"
                args.train_data = args.train_data + "_eng"
            args.test_data = args.test_data + ".json"
            args.train_data = args.train_data + ".json"
    
            dataset = ClipCocoDataset(args.train_data, prefix_length, normalize_prefix=args.normalize_prefix, is_eng=(args.language == "english"), shots_count=args.shots)
            # if args.test_data is None:
            #     with open(args.train_data, 'rb') as f:
            #         all_data = pickle.load(f)
            #         print(all_data)
            #         # test_dataset = [all_data[d] for d in range(len(dataset.prefixes)) if not d in dataset.shots_indexes]
            #         print(dataset.shots_indexes)
            #         print([d for d in range(len(dataset.prefixes)) if not d in dataset.shots_indexes])
            # else:
            #     with open(args.test_data, 'rb') as f:
            #         test_dataset = json.load(f)
            print(dataset.__len__())
            # prefix_dim = 640 if args.is_rn else 512
            
            if args.only_prefix:
                model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                        num_layers=args.num_layers, mapping_type=args.mapping_type, is_eng= (args.language == "english"))
                print("Train only prefix")
            else:
                model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                        num_layers=args.num_layers, mapping_type=args.mapping_type, is_eng= (args.language == "english"))
                print("Train both prefix and GPT")
                sys.stdout.flush()
            if args.model_weights != "":
                print("Using pretrained weights in file: ", args.model_weights)
                model.load_state_dict(torch.load(args.model_weights, map_location= torch.device("cpu")))
            
            model = train(dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix)

            predictor = Predictor(args.out_dir+args.prefix+f"-{(args.epochs-1):03d}.pt", mapping_type=args.mapping_type,clip_length=args.prefix_length_clip, num_layers=args.num_layers,
            is_eng= (args.language == "english"),prefix_length=args.prefix_length, prefix_size=args.prefix_size, clip_model=args.clip_model_type)
            predictions, targets = predictor.test(args.test_data, args.image_path, use_beam_search=True, output_name=args.prefix)
            metrics =evaluate_metrics(predictions, targets, is_eng=(args.language == "english"))
            print(metrics)
            metrics_dict[category] = metrics
        
        with open(f'{args.prefix}_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=4)

    else:
        dataset = ClipCocoDataset(args.train_data, prefix_length, normalize_prefix=args.normalize_prefix, is_eng=(args.language == "english"), shots_count=args.shots)
        # if args.test_data is None:
        #     with open(args.train_data, 'rb') as f:
        #         all_data = pickle.load(f)
        #         print(all_data)
        #         # test_dataset = [all_data[d] for d in range(len(dataset.prefixes)) if not d in dataset.shots_indexes]
        #         print(dataset.shots_indexes)
        #         print([d for d in range(len(dataset.prefixes)) if not d in dataset.shots_indexes])
        # else:
        #     with open(args.test_data, 'rb') as f:
        #         test_dataset = json.load(f)
        print(dataset.__len__())
        # prefix_dim = 640 if args.is_rn else 512
        
        if args.only_prefix:
            model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                    num_layers=args.num_layers, mapping_type=args.mapping_type, is_eng= (args.language == "english"))
            print("Train only prefix")
        else:
            model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                    num_layers=args.num_layers, mapping_type=args.mapping_type, is_eng= (args.language == "english"))
            print("Train both prefix and GPT")
            sys.stdout.flush()
        if args.model_weights != "":
            print("Using pretrained weights in file: ", args.model_weights)
            model.load_state_dict(torch.load(args.model_weights, map_location= torch.device("cpu")))
        
        args.prefix += f'_{args.shots}_{args.language}'
        model = train(dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix)

        predictor = Predictor(args.out_dir+args.prefix+f"-{(args.epochs-1):03d}.pt", mapping_type=args.mapping_type,clip_length=args.prefix_length_clip, num_layers=args.num_layers,
        is_eng= (args.language == "english"),prefix_length=args.prefix_length, prefix_size=args.prefix_size, clip_model=args.clip_model_type)
        predictions, targets = predictor.test(args.test_data, args.image_path, use_beam_search=True, output_name=args.prefix)
        metrics = evaluate_metrics(predictions, targets, is_eng=(args.language == "english"))
        print(metrics)
        with open(f'{args.prefix}_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
