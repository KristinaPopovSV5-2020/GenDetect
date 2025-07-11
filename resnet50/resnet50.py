import numpy as np
import torch
import torch.nn as nn
import timm
import os

from main import calculate_metrics, get_dataloaders

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2
batch_size = 32
epochs = 50
learning_rate = 1e-4
train_dir = '/home/kristina-popov/workspace/AI_Image_Inspector/data/train'
val_dir = '/home/kristina-popov/workspace/AI_Image_Inspector/data/val'


def build_model():
    model = timm.create_model('resnet50', pretrained=True)
    model.fc = nn.Linear(model.get_classifier().in_features, num_classes)
    return model.to(device)


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        if len(labels) == 0: # skip batch with only dummy data
            continue

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability for class 1

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_score = np.array(all_probs)

    return y_true, y_pred, y_score


def train_model():
    train_loader, val_loader = get_dataloaders(train_dir, val_dir, batch_size)
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        y_true, y_pred, y_score = evaluate(model, val_loader)
        print(f"Validation Metrics (Epoch {epoch + 1}):")
        calculate_metrics(y_true, y_pred, y_score)

    torch.save(model.state_dict(), 'finetuned_resnet50.pth')
    print("Model saved as 'finetuned_resnet50.pth'")


if __name__ == '__main__':
    train_model()
