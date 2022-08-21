import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import json
from transformers import AutoTokenizer, CLIPFeatureExtractor
import PIL

prompt = "Retrieve [A] from [Q]"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define hyperparameters
batch_size = 32
num_workers = 1
MAX_LEN = 64 

# a = torch.arange(6)
# print(torch.reshape(a, (3, 2)))

preprocessor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
# print(len(tokenizer.vocab))

def create_dataset_from_csv(path_to_csv):
    # {"id":, "image":, "question":, "answer":}
    with open(path_to_csv, encoding="utf-8") as f:
        data = json.load(f)
    classes = list(set([d["answer"] for d in data]))
    label_to_index = {k: v for v, k in enumerate(classes)}
    for i in range(len(data)):
        data[i]["class"] = label_to_index[data[i]["answer"]]
    return classes, data, label_to_index


classes, data, label_to_index = create_dataset_from_csv("dataset.json")

class VQATrainDataset(Dataset):
    def __init__(self, data, image_folder, prompt):
        # self.labels = labels
        # self.list_IDs = list_IDs
        self.prompt = prompt
        self.data = data
        self.images = [PIL.Image.open(image_folder + x["image"]) for x in self.data]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Select sample
        img = self.images[index]
        img = torch.from_numpy(preprocessor(img, return_tensors='np')["pixel_values"][0])
        q = self.data[index]["question"]
        q = self.prompt.replace("[Q]", q)
        y = self.data[index]["class"]
        q = q.replace("[A]", classes[y])
        tokens = tokenizer(q, return_tensors='pt', padding='max_length',
                                max_length=MAX_LEN, truncation=True)
        # print(tokens.to(device)["input_ids"].shape)
        return {'pixel_values': img.to(device),
            'input_ids': tokens.to(device)["input_ids"][0],
            'attention_mask': tokens.to(device)["attention_mask"][0],
            'labels': torch.tensor(y)}

class VQADataset(Dataset):
    def __init__(self, data, image_folder, prompt):
        # self.labels = labels
        # self.list_IDs = list_IDs
        self.prompt = prompt
        self.data = data
        self.images = [PIL.Image.open(image_folder + x["image"]) for x in self.data]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Select sample
        img = self.images[index]
        q = self.data[index]["question"]
        q = self.prompt.replace("[Q]", q)
        y = self.data[index]["class"]
        return (img, q, y)


def preprocess_data(b):
    # b is the list of tuples of length batch_size
    #   - 0 = img, 
    #   - 1 = question
    #   - 2 = label
    # v = [encode(x[1]) for x in b]
    # out = [torch.nn.functional.pad(t,(0, 0, 0,max_len-len(t)),mode='constant',value=0) for t in v]

    text_list = []
    for _, question, label in b :
        texts = question.replace("[A]", classes[label])
        text_list.append(texts)
    #shape of tokens = (classes * batch_size, padding)
    tokens = tokenizer(text_list, return_tensors='pt', padding='max_length',
                                max_length=MAX_LEN, truncation=True)

    #shape of tokens = (batch_size, classes, padding)
    # tokens = torch.reshape(tokens, (len(b), len(classes), -1))
    

    images = [x[0]  for x in b]
    # print(preprocessor(images, return_tensors='np'))
    images = torch.from_numpy(preprocessor(images, return_tensors='np')["pixel_values"])
        
    return {'pixel_values': images.to(device),
            'input_ids': tokens.to(device)["input_ids"],
            'attention_mask': tokens.to(device)["attention_mask"]} 

def preprocess_data_with_classes(b):
    # b is the list of tuples of length batch_size
    #   - 0 = img, 
    #   - 1 = question
    #   - 2 = label
    # v = [encode(x[1]) for x in b]
    # out = [torch.nn.functional.pad(t,(0, 0, 0,max_len-len(t)),mode='constant',value=0) for t in v]

    text_list = []
    for _, question, _ in b :
        texts = [question.replace("[A]", c) for c in classes]
        text_list.extend(texts)
    #shape of tokens = (classes * batch_size, padding)
    tokens = tokenizer(text_list, return_tensors='pt', padding='max_length',
                                max_length=MAX_LEN, truncation=True)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    #shape of tokens = (batch_size, classes, padding)
    input_ids = torch.reshape(input_ids, (len(b), len(classes), -1))
    attention_mask = torch.reshape(attention_mask, (len(b), len(classes), -1))
    

    images = [x[0]  for x in b]
    # print(preprocessor(images, return_tensors='np'))
    images = torch.from_numpy(preprocessor(images, return_tensors='np')["pixel_values"])
        
    return (images.to(device),
            input_ids.to(device),
            attention_mask.to(device),
            torch.stack([torch.tensor(t[2]) for t in b])) 



# Generators
training_set = VQATrainDataset(data, "./images/",prompt)
test_set = VQADataset(data, "./images/",prompt)
# training_generator = DataLoader(training_set, batch_size=batch_size, collate_fn=preprocess_data, shuffle=True)#, num_workers=num_workers)
test_generator = DataLoader(test_set, batch_size=batch_size, collate_fn=preprocess_data_with_classes, shuffle=True)
i = iter(test_generator)
# img, text, idx = next(i)
# print("img is: \n", img.size())
# print("text is: \n", text)
# print("idx is: \n", idx)
print(next(i))
# validation_set = VQADataset(partition['val'], labels, prompt)
# validation_generator = DataLoader(validation_set, batch_size=batch_size, collate_fn=preprocess_data, num_workers=num_workers)        
