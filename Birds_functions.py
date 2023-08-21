
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import timm
from sklearn.metrics import accuracy_score

def train_model(train_path, valid_path, num_epochs=5, batch_size=32, learning_rate=0.001, model_save_path=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolder(train_path, transform=transform)
    valid_dataset = ImageFolder(valid_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(train_dataset.classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        train_preds, train_targets = [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        train_accuracy = accuracy_score(train_targets, train_preds)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_preds, val_targets = [], []
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_targets, val_preds)
        val_accuracies.append(val_accuracy)

    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
    
    return model, train_accuracies, val_accuracies
