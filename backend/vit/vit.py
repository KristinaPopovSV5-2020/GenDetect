import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import timm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score

from main import get_dataloaders, get_test_loader, plot_confusion_matrix, plot_curves
from env_config import config


@dataclass
class Config:
    batch_size: int = 32
    num_epochs: int = 10
    lr: float = 2e-5
    weight_decay: float = 5e-4
    seed: int = 42
    model_name: str = "vit_small_patch16_224"
    pretrained: bool = True
    dropout: float = 0.5
    patience: int = 2
    output_dir: str = "outputs"
    device: str = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    experiment_mode: str = "finetune_midlayer"   # "finetune", "frozen_midlayer", "finetune_midlayer"
    probe_layer_idx: int = 12
    probe_hidden_dim: int = 256


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FrozenMidLayerProbe(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            num_classes=0
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.probe_layer_idx = cfg.probe_layer_idx

        embed_dim = getattr(self.backbone, "num_features", None)
        if embed_dim is None:
            embed_dim = getattr(self.backbone, "embed_dim")

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, cfg.probe_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.probe_hidden_dim, 2)
        )

    def extract_midlayer_cls(self, x: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone

        x = backbone.patch_embed(x)
        x = backbone._pos_embed(x)

        if hasattr(backbone, "patch_drop"):
            x = backbone.patch_drop(x)
        if hasattr(backbone, "norm_pre"):
            x = backbone.norm_pre(x)

        for i, blk in enumerate(backbone.blocks, start=1):
            x = blk(x)
            if i == self.probe_layer_idx:
                return x[:, 0]

        raise ValueError(f"probe_layer_idx={self.probe_layer_idx} exceeds number of blocks")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cls_feat = self.extract_midlayer_cls(x)
        return self.classifier(cls_feat)


class FineTunedMidLayerProbe(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained,
            num_classes=0
        )

        self.probe_layer_idx = cfg.probe_layer_idx

        embed_dim = getattr(self.backbone, "num_features", None)
        if embed_dim is None:
            embed_dim = getattr(self.backbone, "embed_dim")

        self.classifier = nn.Linear(embed_dim, 2)

    def extract_midlayer_cls(self, x: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone

        x = backbone.patch_embed(x)
        x = backbone._pos_embed(x)

        if hasattr(backbone, "patch_drop"):
            x = backbone.patch_drop(x)
        if hasattr(backbone, "norm_pre"):
            x = backbone.norm_pre(x)

        for i, blk in enumerate(backbone.blocks, start=1):
            x = blk(x)
            if i == self.probe_layer_idx:
                return x[:, 0]

        raise ValueError(f"probe_layer_idx={self.probe_layer_idx} exceeds number of blocks")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_feat = self.extract_midlayer_cls(x)
        return self.classifier(cls_feat)


def build_model(cfg: Config):
    if cfg.experiment_mode == "frozen_midlayer":
        return FrozenMidLayerProbe(cfg)

    if cfg.experiment_mode == "finetune_midlayer":
        return FineTunedMidLayerProbe(cfg)

    return timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=2,
        drop_rate=cfg.dropout
    )


def compute_metrics(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(y_true, y_score) if y_score is not None else None

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        probs = torch.softmax(logits, dim=1)
        fake_probs = probs[:, 1]
        preds = (fake_probs >= threshold).long()

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(fake_probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = epoch_loss
    metrics["y_true"] = all_labels
    metrics["y_pred"] = all_preds
    metrics["y_score"] = all_probs
    metrics["threshold"] = threshold

    return metrics


def find_best_f1_threshold(y_true, y_prob, num_thresholds=201):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    best_threshold = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    return best_threshold, best_f1


def print_threshold_metrics(y_true, y_score, threshold, name):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    print(
        f"{name} threshold={threshold:.3f} | "
        f"acc={acc:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
    )


def load_trained_model(checkpoint_path: str, cfg: Config):
    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(cfg.device)
    model.eval()
    return model, checkpoint

def save_results_with_paths(dataset, y_true, y_pred, file_path="result.txt"):
    TP, FP, TN, FN = [], [], [], []

    if not hasattr(dataset, "samples"):
        raise ValueError("Dataset does not expose .samples, so paths cannot be recovered.")

    paths_and_labels = dataset.samples

    for (path, _label), yt, yp in zip(paths_and_labels, y_true, y_pred):
        if yt == 1 and yp == 1:
            TP.append(path)
        elif yt == 0 and yp == 1:
            FP.append(path)
        elif yt == 0 and yp == 0:
            TN.append(path)
        elif yt == 1 and yp == 0:
            FN.append(path)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("TP:\n" + "\n".join(TP) + "\n\n")
        f.write("FP:\n" + "\n".join(FP) + "\n\n")
        f.write("TN:\n" + "\n".join(TN) + "\n\n")
        f.write("FN:\n" + "\n".join(FN) + "\n\n")


def main(run_training=True, best_threshold=0.5):
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("Experiment mode:", cfg.experiment_mode)
    if cfg.experiment_mode in ["frozen_midlayer", "finetune_midlayer"]:
        print("Probe layer:", cfg.probe_layer_idx)

    train_loader, val_loader = get_dataloaders(
        config.TRAIN_DIR,
        config.VAL_DIR,
        cfg.batch_size
    )

    test_loader = get_test_loader(
        config.TEST_DIR,
        cfg.batch_size
    )

    model = build_model(cfg).to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    best_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    best_y_trues = None
    best_y_scores = None

    print("Train size:", len(train_loader.dataset))
    print("Val size:", len(val_loader.dataset))
    print("Test size:", len(test_loader.dataset))

    if run_training:
        for epoch in range(cfg.num_epochs):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
            val_metrics = evaluate(model, val_loader, criterion, cfg.device, threshold=0.5)

            print(f"\nEpoch [{epoch + 1}/{cfg.num_epochs}]")
            print(
                f"Train - loss: {train_metrics['loss']:.4f}, "
                f"acc: {train_metrics['accuracy']:.4f}, "
                f"precision: {train_metrics['precision']:.4f}, "
                f"recall: {train_metrics['recall']:.4f}, "
                f"f1: {train_metrics['f1']:.4f}"
            )
            print(
                f"Val   - loss: {val_metrics['loss']:.4f}, "
                f"acc: {val_metrics['accuracy']:.4f}, "
                f"precision: {val_metrics['precision']:.4f}, "
                f"recall: {val_metrics['recall']:.4f}, "
                f"f1: {val_metrics['f1']:.4f}, "
                f"roc_auc: {val_metrics['roc_auc']:.4f}"
            )

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                best_y_trues = val_metrics["y_true"]
                best_y_scores = val_metrics["y_score"]

                checkpoint_path = os.path.join(cfg.output_dir, "best_vit_model.pt")
                torch.save({
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_f1": best_f1,
                    "config": cfg.__dict__,
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= cfg.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        print(f"\nBest validation F1: {best_f1:.4f} at epoch {best_epoch}")
        print_threshold_metrics(best_y_trues, best_y_scores, 0.5, "Validation default")

        best_threshold, tuned_f1 = find_best_f1_threshold(best_y_trues, best_y_scores)
        print(f"Best threshold by validation F1: {best_threshold:.3f}")
        print(f"Best validation F1 at tuned threshold: {tuned_f1:.4f}")

    checkpoint_path = os.path.join(cfg.output_dir, "best_vit_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(cfg.device)
    model.eval()

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        cfg.device,
        threshold=best_threshold,
    )

    save_results_with_paths(test_loader.dataset, test_metrics["y_true"], test_metrics["y_pred"], file_path=os.path.join("result.txt"))

    print("\nTest metrics (using validation-tuned threshold)")
    print(
        f"Test  - loss: {test_metrics['loss']:.4f}, "
        f"acc: {test_metrics['accuracy']:.4f}, "
        f"precision: {test_metrics['precision']:.4f}, "
        f"recall: {test_metrics['recall']:.4f}, "
        f"f1: {test_metrics['f1']:.4f}, "
        f"roc_auc: {test_metrics['roc_auc']:.4f}"
    )

    plot_confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"])
    plot_curves(test_metrics["y_true"], test_metrics["y_score"])


if __name__ == "__main__":
    main()
    # main(False, 0.5)