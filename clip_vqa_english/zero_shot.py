import argparse
from ast import arg
import torch
import sys
import json
import pickle
from clipmodel import get_model
from clipdataset import create_test_generator, create_train_dataset
from cliptest import test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', default='./dataset.json')
    parser.add_argument('--image_path', default='./images/')
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
    parser.add_argument('--model_weights', default="openai/clip-vit-base-patch32")
    parser.add_argument('--clip_model_type', default="RN50x4", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    

    model = get_model(args.model_weights)
    classes, label_to_index, test_generator, data = create_test_generator(args.test_data, args.image_path, args.bs)
    test(model, test_generator, classes, data, label_to_index)

#     prefix_length = args.prefix_length
#     dataset = ClipCocoDataset(args.train_data, prefix_length, normalize_prefix=args.normalize_prefix, is_eng=(args.language == "english"), shots_count=args.shots)
#     # if args.test_data is None:
#     #     with open(args.train_data, 'rb') as f:
#     #         all_data = pickle.load(f)
#     #         print(all_data)
#     #         # test_dataset = [all_data[d] for d in range(len(dataset.prefixes)) if not d in dataset.shots_indexes]
#     #         print(dataset.shots_indexes)
#     #         print([d for d in range(len(dataset.prefixes)) if not d in dataset.shots_indexes])
#     # else:
#     #     with open(args.test_data, 'rb') as f:
#     #         test_dataset = json.load(f)
#     print(dataset.__len__())
#     # prefix_dim = 640 if args.is_rn else 512
#     prefix_dim = args.prefix_size
#     args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
#     if args.only_prefix:
#         model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
#                                   num_layers=args.num_layers, mapping_type=args.mapping_type, is_eng= (args.language == "english"))
#         print("Train only prefix")
#     else:
#         model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
#                                   num_layers=args.num_layers, mapping_type=args.mapping_type, is_eng= (args.language == "english"))
#         print("Train both prefix and GPT")
#         sys.stdout.flush()
#     if args.model_weights != "":
#         print("Using pretrained weights in file: ", args.model_weights)
#         model.load_state_dict(torch.load(args.model_weights, map_location= torch.device("cpu")))
    
#     args.prefix += f'_{args.shots}_{args.language}'
#     model = train(dataset, model, args, output_dir=args.out_dir, output_prefix=args.prefix)

#     predictor = Predictor(args.out_dir+args.prefix+f"-{(args.epochs-1):03d}.pt", mapping_type=args.mapping_type,clip_length=args.prefix_length_clip, num_layers=args.num_layers,
#      is_eng= (args.language == "english"),prefix_length=args.prefix_length, prefix_size=args.prefix_size, clip_model=args.clip_model_type)
#     predictions, targets = predictor.test(args.test_data, args.image_path, use_beam_search=True)
#     print(evaluate_metrics(predictions, targets, is_eng=(args.language == "english")))


if __name__ == '__main__':
    main()
