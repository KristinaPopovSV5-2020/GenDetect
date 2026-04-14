import numpy as np
import random
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import timm
from main import get_test_loader, calculate_metrics, plot_confusion_matrix, plot_curves, \
    get_dataloaders
from env_config import config as env_config
from resnet50.train_config import TrainResNetConfig

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 1
batch_size = 32
epochs = 20

class ResNetClassifier(nn.Module):
    def __init__(self, train_config: TrainResNetConfig):
        super().__init__()

        self.train_config = train_config

        self.backbone = timm.create_model(
            'resnet50',
            pretrained=True,
            num_classes=0
        )

        # freeze all
        for param in self.backbone.parameters():
            param.requires_grad = False

        # unfreeze selected layers
        for name, param in self.backbone.named_parameters():
            if any(layer in name for layer in train_config.unfreeze_layers):
                param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(train_config.dropout),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)



def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    all_probs = []
    all_labels = []

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images).view(-1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        all_probs.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    train_auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, train_auc



def evaluate(model, loader, threshold):
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs >= threshold).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)



def train_model(resnet_config: TrainResNetConfig):
    train_loader, val_loader = get_dataloaders(
        env_config.TRAIN_DIR,
        env_config.VAL_DIR,
        batch_size
    )

    model = ResNetClassifier(resnet_config).to(device)
    criterion = nn.BCEWithLogitsLoss()

    backbone_params = [
        p for n, p in model.backbone.named_parameters()
        if any(layer in n for layer in resnet_config.unfreeze_layers)
    ]

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4},
    ])

    scheduler = None
    if resnet_config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.3,
            patience=2
        )

    best_auc = 0.0

    for epoch in range(epochs):
        avg_loss, train_auc = train_one_epoch(model, train_loader, criterion, optimizer)

        y_true, y_pred, y_score = evaluate(model, val_loader, resnet_config.threshold)
        val_auc = roc_auc_score(y_true, y_score)

        print(f"Epoch [{epoch + 1}/{epochs}]")
        print("-------------------------------")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Train AUC:  {train_auc:.4f}")
        print(f"Val AUC:    {val_auc:.4f}")
        print(f"AUC Gap:    {train_auc - val_auc:.4f}")

        print("Validation Metrics:")
        calculate_metrics(y_true, y_pred, y_score)

        if scheduler is not None:
            scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), resnet_config.model_name)



def load_trained_model(resnet_config: TrainResNetConfig):
    model = ResNetClassifier(resnet_config).to(device)
    model.load_state_dict(torch.load(resnet_config.model_name, map_location=device))
    model.eval()
    return model


def test_model(test_dir, resnet_config: TrainResNetConfig):
    model = load_trained_model(resnet_config)
    test_loader = get_test_loader(test_dir, batch_size)

    y_true, y_pred, y_score = evaluate(
        model,
        test_loader,
        resnet_config.threshold
    )

    print("Test Set Metrics:")
    calculate_metrics(y_true, y_pred, y_score)
    plot_confusion_matrix(y_true, y_pred)
    plot_curves(y_true, y_score)


config1 = TrainResNetConfig(
    unfreeze_layers=("layer4",),
    dropout=0.5,
    threshold=0.5,
    use_scheduler=False,
    model_name="finetuned_resnet50-v2.pth"
)
config2 = TrainResNetConfig(
    unfreeze_layers=("layer3", "layer4"),
    dropout=0.6,
    threshold=0.45,
    use_scheduler=True,
    model_name="best_model-v2.pth"
)
config3 = TrainResNetConfig(
    unfreeze_layers=("layer3", "layer4"),
    dropout=0.4,
    threshold=0.45,
    use_scheduler=True,
    model_name="config3_v2.pth"
)

if __name__ == '__main__':
    #train_model(config3)
    test_model(
        env_config.TEST_DIR,
        config3
    )
