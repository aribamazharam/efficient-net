
import torch
import torchvision
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.optim as optim

def train_yoga_model(train_path, valid_path, num_epochs=10, batch_size=64, learning_rate=0.001, model_save_path=None):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(train_path, transform=train_transforms)
    val_data = datasets.ImageFolder(valid_path, transform=val_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, len(train_data.classes))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_prec': [],
        'train_recall': [],
        'val_loss': [],
        'val_acc': [],
        'val_prec': [],
        'val_recall': []
    }

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        all_labels = []
        all_preds = []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        epoch_loss /= len(train_data)
        
        # Evaluation logic will be called here...

    if model_save_path:
        torch.save(model.state_dict(), model_save_path)

    return model, history
