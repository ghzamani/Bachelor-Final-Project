import torch
from transformers import TrainingArguments
from clipfamodel import get_model
from transformers import Trainer
from torch.cuda.amp import autocast
import multiprocessing
import gc
import os
from clipfadataset import prompt
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def test(model, generator, classes, data, label_to_index):
    model = model.to(device)
    accuracy = 0
    with torch.no_grad():
        for images, texts, masks, labels, index in tqdm(generator):
            image_features = model.get_image_features(pixel_values = images)
            predictions = []
            print()
            for i in range(images.shape[0]):
                image_feature = image_features[i].unsqueeze(0)
                class_features = model.get_text_features(input_ids= texts[i], attention_mask=masks[i])
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                class_features /= class_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_feature @ class_features.T).softmax(dim=-1)
                # print("similarity: ", similarity)
                predictions.append(torch.argmax(similarity[0]))
            
            predictions = torch.tensor(predictions)
            for i in range(len(labels)):
                print("Image:", data[index[i]]["image"])
                print("true prompt:", prompt.replace("[Q]", data[index[i]]["question"]).replace("[A]", classes[labels[i]]))
                print("pred prompt:", prompt.replace("[Q]", data[index[i]]["question"]).replace("[A]", classes[predictions[i]]))
            # print("labels: ", labels)
            # print("predictions: ", predictions)
            correct_answers = torch.sum((labels == predictions))
            accuracy += correct_answers
    accuracy = accuracy / len(generator.dataset)
    print(label_to_index)
    print("accuracy: ", accuracy)