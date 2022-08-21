import os
import clip
import torch
from torchvision.datasets import CIFAR100

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
classes = [] 

def download_dataset(path):
    # Download the dataset
    #[(image, class_id)] --> for vqa [(image, question, class_id)]
    global classes

    dataset = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    return dataset

def preprocess_data(dataset, prompt, index):
    # prompt format e.g. Retrieve [A] from [Q]

    # Prepare the inputs
    image, question, class_id = dataset[index]
    image_input = preprocess(image).unsqueeze(0).to(device)
    # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
    prompt = prompt.replace("[Q]", question)
    text_inputs = torch.cat([clip.tokenize(prompt.replace("[A]", c)) for c in classes]).to(device)
    return (text_inputs, image_input, class_id)

def predict_topk(model, text_inputs, image_input, k):
    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(k)
    return values, indices

def test(model, text_inputs, image_input, class_id, k):
    values, indices = predict_topk(model, text_inputs, image_input, k)
    # Print the result
    print("True Label:", classes[class_id])
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")