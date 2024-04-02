import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader

from tools.model import *

from tools.dataset import create_dataset

test_data_path = 'root/test'
saved_model_path = 'pretrained/test_model_weights.pth'

batch_size = 32
test_dataset = create_dataset(test_data_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

model = ResNet(Bottleneck, [3, 4, 23, 3], nb_classes=32, channel=3, pretrained=True)


# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model (make sure the class definition is accessible)
model = ResNet(Bottleneck, [3, 4, 23, 3], nb_classes=32, channel=3, pretrained=False) 
model.to(device)

# Load the saved model checkpoint
checkpoint = torch.load(saved_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Set model to evaluation mode
model.eval()

# Iterate over test dataset for inference
top1_correct = 0
top5_correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        # Top-1 accuracy
        top1_correct += (predicted == labels).sum().item()

        # Top-5 accuracy
        _, top5_predicted = torch.topk(outputs, 5, dim=1)
        top5_correct += torch.sum(top5_predicted == labels.view(-1, 1)).item()

        total += labels.size(0)

# Compute accuracy
top1_accuracy = top1_correct / total
top5_accuracy = top5_correct / total

print(f"Top-1 Accuracy on test set: {top1_accuracy * 100:.2f}%")
print(f"Top-5 Accuracy on test set: {top5_accuracy * 100:.2f}%")