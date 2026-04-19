import os
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from unittest import result

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from main import plot_confusion_matrix, plot_curves

import timm
import csv

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from env_config import config

@dataclass
class Config:
    data_dir: str = config.INPUT_DATA_FOLDER
    test_dir: str = config.OUTPUT_DATASET_FOLDER
    image_size: int = 224
    batch_size: int = 32
    num_epochs: int = 10
    lr: float = 2e-5
    weight_decay: float = 5e-4
    val_split: float = 0.1
    test_split: float = 0.2
    num_workers: int = 0
    seed: int = 42
    model_name: str = "vit_small_patch16_224"
    pretrained: bool = True
    dropout: float = 0.5
    patience: int = 2
    output_dir: str = "outputs"
    device: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    experiment_mode: str = "finetune_midlayer"   # possible modes: "finetune", "frozen_midlayer", "finetune_midlayer"
    probe_layer_idx: int = 12            
    probe_hidden_dim: int = 256         
    max_samples_per_class: int = 8000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class BinaryImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, max_samples: int = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.max_samples = max_samples

        self.class_to_idx = {"real": 0, "fake": 1}
        self.samples = []

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for class_name, label in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")

            class_count = 0

            for file_path in class_dir.rglob("*"):
                if file_path.suffix.lower() in valid_extensions:
                    self.samples.append((str(file_path), label))
                    class_count += 1

                    if self.max_samples is not None and class_count >= self.max_samples:
                        break

        if not self.samples:
            raise ValueError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path)
        if image.mode == "P":
            image = image.convert("RGBA")
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def build_transforms(cfg: Config):
    train_tfms = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    return train_tfms, val_tfms

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
                return x[:, 0]   # CLS

        raise ValueError(f"probe_layer_idx={self.probe_layer_idx} exceeds number of blocks")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cls_feat = self.extract_midlayer_cls(x)
        logits = self.classifier(cls_feat)
        return logits
    

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
        logits = self.classifier(cls_feat)
        return logits

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

def save_wrong_predictions_csv(paths, output_csv_path):
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path"])
        for path in paths:
            writer.writerow([path])

@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5, collect_wrong_paths=False):
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []
    wrong_paths = []

    for batch in loader:
        if len(batch) == 3:
            images, labels, paths = batch
        else:
            images, labels = batch
            paths = None

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        probs = torch.softmax(logits, dim=1)
        fake_probs = probs[:, 1]
        preds = (fake_probs >= threshold).long()

        preds_cpu = preds.detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
        probs_cpu = fake_probs.detach().cpu().numpy()

        all_preds.extend(preds_cpu)
        all_labels.extend(labels_cpu)
        all_probs.extend(probs_cpu)

        if collect_wrong_paths and paths is not None:
            for pred, true, path in zip(preds_cpu, labels_cpu, paths):
                if pred != true:
                    wrong_paths.append(path)

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    metrics["loss"] = epoch_loss
    metrics["y_true"] = all_labels
    metrics["y_pred"] = all_preds
    metrics["y_score"] = all_probs
    metrics["threshold"] = threshold
    metrics["wrong_paths"] = wrong_paths

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

def predict_image(
    image_path: str,
    model,
    cfg: Config,
    threshold: float = 0.5
):
    _, val_tfms = build_transforms(cfg)

    image = Image.open(image_path)
    if image.mode == "P":
        image = image.convert("RGBA")
    image = image.convert("RGB")

    image_tensor = val_tfms(image).unsqueeze(0).to(cfg.device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    real_prob = float(probs[0].item())
    fake_prob = float(probs[1].item())

    pred_idx = 1 if fake_prob >= threshold else 0
    pred_label = "fake" if pred_idx == 1 else "real"

    return {
        "image_path": image_path,
        "predicted_class_idx": pred_idx,
        "predicted_label": pred_label,
        "prob_real": real_prob,
        "prob_fake": fake_prob,
        "threshold": threshold,
    }


def show_image(image_path: str):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis("off")
    plt.title(image_path)
    plt.show()

def print_prediction(result: dict):
    print("\nInference result")
    print(f"Image: {result['image_path']}")
    print(f"Predicted label: {result['predicted_label']}")
    print(f"P(real): {result['prob_real']:.4f}")
    print(f"P(fake): {result['prob_fake']:.4f}")
    print(f"Threshold used: {result['threshold']:.3f}")
    show_image(result["image_path"])



def run_inference_example():
    cfg = Config()

    checkpoint_path = os.path.join(cfg.output_dir, "best_vit_model.pt")
    model, checkpoint = load_trained_model(checkpoint_path, cfg)

    threshold = checkpoint.get("best_threshold", 0.5)

    image_path = f"{cfg.test_dir}/fake/4800.jpg"
    result = predict_image(image_path, model, cfg, threshold=threshold)
    print_prediction(result)


def main(run_training=True, best_threshold=0.5):
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("Experiment mode:", cfg.experiment_mode)
    if cfg.experiment_mode == "frozen_midlayer":
        print("Probe layer:", cfg.probe_layer_idx)

    train_tfms, val_tfms = build_transforms(cfg)

    full_dataset = BinaryImageFolderDataset(
        cfg.data_dir,
        transform=None,
        max_samples=cfg.max_samples_per_class
    )


    labels = [label for _, label in full_dataset.samples]
    indices = np.arange(len(full_dataset))

    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices,
        labels,
        test_size=cfg.val_split + cfg.test_split,
        random_state=cfg.seed,
        stratify=labels
    )

    val_ratio_within_temp = cfg.val_split / (cfg.val_split + cfg.test_split)

    val_indices, test_indices, _, _ = train_test_split(
        temp_indices,
        temp_labels,
        test_size=1 - val_ratio_within_temp,
        random_state=cfg.seed,
        stratify=temp_labels
    )

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)

    class TransformedSubset(Dataset):
        def __init__(self, subset, transform, return_path: bool = False):
            self.subset = subset
            self.transform = transform
            self.return_path = return_path

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            original_idx = self.subset.indices[idx]
            img_path, label = self.subset.dataset.samples[original_idx]

            image = Image.open(img_path)
            if image.mode == "P":
                image = image.convert("RGBA")
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

            if self.return_path:
                return image, label, img_path

            return image, label

    train_dataset = TransformedSubset(train_subset, train_tfms, return_path=False)
    val_dataset = TransformedSubset(val_subset, val_tfms, return_path=False)
    test_dataset = TransformedSubset(test_subset, val_tfms, return_path=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False
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

    best_y_preds = None
    best_y_trues = None

    print("Dataset size:", len(full_dataset))
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))

    start_time = time.time()

    if run_training:
        for epoch in range(cfg.num_epochs):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
            val_metrics = evaluate(model, val_loader, criterion, cfg.device)

            print(f"\nEpoch [{epoch + 1}/{cfg.num_epochs}]")
            print(
                f"Train - loss: {train_metrics['loss']:.4f}, "
                f"acc: {train_metrics['accuracy']:.4f}, "
                f"precision: {train_metrics['precision']:.4f}, "
                f"recall: {train_metrics['recall']:.4f}, "
                f"f1: {train_metrics['f1']:.4f}, "
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
                best_y_preds = val_metrics["y_pred"]
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

        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTotal training time: {total_time/60:.2f} minutes")

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
        collect_wrong_paths=True
    )

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
    # main(False, 0.585)
    # run_inference_example()