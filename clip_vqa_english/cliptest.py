import torch
from transformers import TrainingArguments
from clipmodel import get_model
from transformers import Trainer
from torch.cuda.amp import autocast
import multiprocessing
import gc
import os
from clipdataset import test_generator, classes, label_to_index#, test_ds
from tqdm import tqdm

def test(model, generator):
    accuracy = 0
    with torch.no_grad():
        for images, texts, masks, labels in tqdm(generator):
            image_features = model.get_image_features(pixel_values = images)
            predictions = []
            for i in range(images.shape[0]):
                image_feature = image_features[i].unsqueeze(0)
                class_features = model.get_text_features(input_ids= texts[i], attention_mask=masks[i])
                image_feature /= image_feature.norm(dim=-1, keepdim=True)
                class_features /= class_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_feature @ class_features.T).softmax(dim=-1)
                print("similarity: ", similarity)
                predictions.append(torch.argmax(similarity[0]))
            
            predictions = torch.tensor(predictions)
            print("labels: ", labels)
            print("predictions: ", predictions)
            print(classes)
            print(label_to_index)
            correct_answers = torch.sum((labels == predictions))
            accuracy += correct_answers
    accuracy = accuracy / len(generator.dataset)
    print(accuracy)


test(get_model(), test_generator)