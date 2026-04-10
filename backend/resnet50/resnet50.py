import numpy as np
from timm.layers import config
import torch
import torch.nn as nn
import timm
import cv2
from main import get_test_loader, calculate_metrics, plot_confusion_matrix, plot_curves, \
    get_dataloaders

from ..env_config import config

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2
batch_size = 32
epochs = 50
learning_rate = 1e-4



def build_model():
    model = timm.create_model('resnet50', pretrained=True, num_classes=0)  # without FC dense

    # Freeze ResNet parameters
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    full_model = nn.Sequential(
        model,
        classifier
    )

    return full_model.to(device)


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # convert to float and add dimension

        if len(labels) == 0:
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

            probs = outputs.squeeze(1)  # [batch_size]
            preds = (probs >= 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_score = np.array(all_probs)

    return y_true, y_pred, y_score



def train_model():
    train_loader, val_loader = get_dataloaders(config.TRAIN_DIR, config.VAL_DIR, batch_size)
    model = build_model()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        y_true, y_pred, y_score = evaluate(model, val_loader)
        print(f"Validation Metrics (Epoch {epoch + 1}):")
        calculate_metrics(y_true, y_pred, y_score)

    torch.save(model.state_dict(), 'finetuned_resnet50.pth')
    print("Model saved as 'finetuned_resnet50.pth'")


def load_trained_model(model_path):
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def test_model(model_path, test_dir, batch_size=32):
    model = load_trained_model(model_path)
    test_loader = get_test_loader(test_dir, batch_size)

    y_true, y_pred, y_score = evaluate(model, test_loader)

    print("Test Set Metrics:")
    calculate_metrics(y_true, y_pred, y_score)
    plot_confusion_matrix(y_true, y_pred)
    plot_curves(y_true, y_score)


def visualize_heatmap(img, heatmap ):
    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB format and apply colormap
    heatmap = cv2.applyColorMap(
        np.array(255 * heatmap, dtype=np.uint8),
        cv2.COLORMAP_JET
    )

    # Overlay the heatmap on the original image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    cv2.imshow('Grad-CAM', superimposed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #train_model()
    test_model('finetuned_resnet50.pth', config.TEST_DIR)
