import os

import torch
from torchvision import datasets, transforms

def create_dataset(data_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),           # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])

    dataset = datasets.ImageFolder(root=os.path.join(data_path), transform=transform)

    return dataset
