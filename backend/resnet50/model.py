import numpy as np
from itertools import product
import random
from sklearn.metrics import roc_auc_score, f1_score
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

search_epochs = 6   # fast search
patience = 3 # for early stopping

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



def train_model(resnet_config: TrainResNetConfig, epochs):
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
    best_threshold = 0.5
    patience_counter = 0

    for epoch in range(epochs):
        avg_loss, train_auc = train_one_epoch(model, train_loader, criterion, optimizer)

        y_true, _, y_score = evaluate(model, val_loader, 0.5)

        val_auc = roc_auc_score(y_true, y_score)

        current_t, current_f1 = find_best_threshold(y_true, y_score)
        y_pred = (y_score >= current_t).astype(int)

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
            patience_counter = 0  # reset
            best_threshold = current_t
            torch.save({
                "model_state": model.state_dict(),
                "threshold": best_threshold
            }, resnet_config.model_name)
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return best_auc



def load_trained_model(resnet_config: TrainResNetConfig):
    checkpoint = torch.load(
        resnet_config.model_name,
        map_location=device,
        weights_only=False
    )

    model = ResNetClassifier(resnet_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, checkpoint["threshold"]


def test_model(test_dir, resnet_config: TrainResNetConfig):
    model, threshold = load_trained_model(resnet_config)
    test_loader = get_test_loader(test_dir, batch_size)

    y_true, y_pred, y_score = evaluate(
        model,
        test_loader,
        threshold
    )

    print("Test Set Metrics:")
    calculate_metrics(y_true, y_pred, y_score)
    plot_confusion_matrix(y_true, y_pred)
    plot_curves(y_true, y_score)


def find_best_threshold(y_true, y_score):
    best_t, best_f1 = 0.5, 0

    for t in np.linspace(0.3, 0.7, 9):
        preds = (y_score >= t).astype(int)
        f1 = f1_score(y_true, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1

def generate_configs():
    unfreeze_options = [
        ("layer4",),
        ("layer3", "layer4")
    ]

    dropouts = [0.4, 0.6]
    schedulers = [True, False]

    configs = []

    for u, d, s in product(unfreeze_options, dropouts, schedulers):
        name = f"model_{'_'.join(u)}_{d}_{s}.pth"
        configs.append(
            TrainResNetConfig(u, d, 0.5, s, name)
        )

    return configs


def run_search():
    configs = generate_configs()
    results = []

    for i, config in enumerate(configs):
        print(f"----Training config {i+1}/{len(configs)} ------")
        print(config)

        val_auc = train_model(config, search_epochs)
        results.append((config, val_auc))

    best_config, best_auc = max(results, key=lambda x: x[1])

    print("-----------------")
    print(f"Best config: {best_config}")
    print(f"Best AUC: {best_auc}")

    return best_config


if __name__ == '__main__':
    #best_config = run_search()
    best_config = TrainResNetConfig(unfreeze_layers=("layer3", "layer4"),dropout=0.6, use_scheduler=False, model_name='resnet50-best-config.pth')
    #train_model(best_config, epochs=epochs)
    test_model(
        env_config.TEST_DIR,
        best_config
    )
