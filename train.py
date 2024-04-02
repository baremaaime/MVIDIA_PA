import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader

from tools.model import *

from tools.dataset import create_dataset

train_data_path = 'root/train'
saved_model_path = 'pretrained/test_model_weights.pth'

batch_size = 32
train_dataset = create_dataset(train_data_path)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = ResNet(Bottleneck, [3, 4, 23, 3], nb_classes=32, channel=3, pretrained=True)

pretrained_weights = torch.load('model_best.pth.tar')
model_dict = model.state_dict()
pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_dict}

model_dict.update(pretrained_weights)
model.load_state_dict(model_dict)

batch_size = 32
learning_rate = 0.001
num_epochs = 15

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
     
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {100 * epoch_accuracy:.2f}%')
    
    if epoch % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
        }, saved_model_path)