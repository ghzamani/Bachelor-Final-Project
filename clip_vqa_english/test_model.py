import torch
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re
import clip

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

classes = []
prompt = "Retrieve [A] from [Q]"

# Define hyperparameters
batch_size = 32
num_workers = 1

class VQADataset(Dataset):
    def __init__(self, list_IDs, labels, prompt):
        self.labels = labels
        self.list_IDs = list_IDs
        self.prompt = prompt
    
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]
        img = q_img[ID][0]
        q = q_img[ID][1]
        q = self.prompt.replace("[Q]", q)
        y = index
        return (img, q, y)

def preprocess_data(b):
    # b is the list of tuples of length batch_size
    #   - 0 = img, 
    #   - 1 = question
    #   - 2 = label
    # v = [encode(x[1]) for x in b]
    # out = [torch.nn.functional.pad(t,(0, 0, 0,max_len-len(t)),mode='constant',value=0) for t in v]
    text_features = []
    for _, question, _ in b :
        text_inputs = torch.cat([clip.tokenize(question.replace("[A]", c)) for c in classes]).to(device)
        text_features.append(text_inputs)
        
    img = [torch.tensor(preprocess(x[0])) for x in b]
        
    return (torch.stack(img),
            torch.stack(text_features),
            torch.stack([int(t[2]) for t in b]))
            # torch.LongTensor([int(t[2]) for t in b])) 

# Generators
training_set = VQADataset(partition['train'], labels, prompt)
training_generator = DataLoader(training_set, batch_size=batch_size, collate_fn=preprocess_data, shuffle=True, num_workers=num_workers)

validation_set = VQADataset(partition['val'], labels, prompt)
validation_generator = DataLoader(validation_set, batch_size=batch_size, collate_fn=preprocess_data, num_workers=num_workers)        


def test(model, generator):
    with torch.no_grad():
        for images, questions, labels in tqdm(generator):
            continue
            # features = model.encode_image(images.to(device))

            # all_features.append(features)
            # all_labels.append(labels)