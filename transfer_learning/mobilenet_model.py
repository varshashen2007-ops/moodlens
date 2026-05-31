# ================================================================
# MoodLens v2 — transfer_learning/mobilenet_model.py
#
# ================================================================

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader


# allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_loader import EMOTION_LABELS


class EmotionHead(nn.Module):
    def __init__(self, in_features=1280, num_classes=7, dropout=0.4):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),

            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.head(x)


class MoodNetV2(nn.Module):
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()

        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

        # IMPORTANT:
        # Do NOT modify first conv layer.
        # We now feed 3-channel 224x224 images, exactly like ImageNet.
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = EmotionHead(
            in_features=1280,
            num_classes=num_classes,
            dropout=dropout
        )

        self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  [MoodNetV2] Backbone frozen — training head only")

    def unfreeze_top_blocks(self, num_blocks=4):
        total_blocks = len(self.backbone)

        for i, block in enumerate(self.backbone):
            if i >= total_blocks - num_blocks:
                for param in block.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [MoodNetV2] Unfroze last {num_blocks} blocks")
        print(f"  Trainable params now: {trainable:,}")

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.pool(features)
        flat = pooled.flatten(1)
        logits = self.head(flat)
        return logits

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        return pred.item(), conf.item(), probs.squeeze()

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def build_mobilenet_dataloaders(data_dir="data/", batch_size=32):
    """
    MobileNetV2 expects:
      - 3 channels
      - 224x224 size
      - ImageNet normalization
    """

    train_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),
            scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=train_tfms
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=val_tfms
    )

    class_counts = torch.zeros(7)

    for _, label in train_dataset.samples:
        class_counts[label] += 1

    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, class_weights


def train_transfer(data_dir="data/", device_str="auto"):
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print("\n" + "=" * 65)
    print("  MoodLens v2 — MobileNetV2 Transfer Learning")
    print(f"  Device: {device}")
    print("=" * 65)

    train_loader, val_loader, class_weights = build_mobilenet_dataloaders(
        data_dir=data_dir,
        batch_size=32
    )

    model = MoodNetV2(num_classes=7, dropout=0.4).to(device)

    total, trainable = model.count_params()
    print(f"\n  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.05
    )

    os.makedirs("models", exist_ok=True)

    history = []
    best_acc = 0.0

    def run_epoch(loader, optimizer=None):
        is_train = optimizer is not None
        model.train() if is_train else model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        context = torch.enable_grad() if is_train else torch.no_grad()

        with context:
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                if is_train:
                    optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        return total_loss / total, correct / total

    # ==========================================================
    # PHASE 1 — Train classifier head only
    # ==========================================================
    print("\n── Phase 1: Feature extraction — backbone frozen ──\n")

    model.freeze_backbone()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10
    )

    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}")
    print("  " + "-" * 55)

    for epoch in range(1, 11):
        train_loss, train_acc = run_epoch(train_loader, optimizer)
        val_loss, val_acc = run_epoch(val_loader)
        scheduler.step()

        flag = ""

        if val_acc > best_acc:
            best_acc = val_acc
            flag = "  ✓ best"

            torch.save({
                "epoch": epoch,
                "phase": 1,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "config": {
                    "dropout": 0.4,
                    "input_size": 224,
                    "channels": 3,
                    "model": "mobilenet_v2"
                }
            }, "models/mobilenet_best.pt")

        history.append({
            "epoch": epoch,
            "phase": 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        print(
            f"  {epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
            f"{val_loss:>9.4f}  {val_acc:>7.2%}{flag}"
        )

    # ==========================================================
    # PHASE 2 — Fine-tune final MobileNet blocks
    # ==========================================================
    print("\n── Phase 2: Fine-tuning — last blocks unfrozen ──\n")

    model.unfreeze_top_blocks(num_blocks=4)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=20,
        eta_min=1e-6
    )

    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}")
    print("  " + "-" * 55)

    for local_epoch in range(1, 21):
        epoch = local_epoch + 10

        train_loss, train_acc = run_epoch(train_loader, optimizer)
        val_loss, val_acc = run_epoch(val_loader)
        scheduler.step()

        flag = ""

        if val_acc > best_acc:
            best_acc = val_acc
            flag = "  ✓ best"

            torch.save({
                "epoch": epoch,
                "phase": 2,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "config": {
                    "dropout": 0.4,
                    "input_size": 224,
                    "channels": 3,
                    "model": "mobilenet_v2"
                }
            }, "models/mobilenet_best.pt")

        history.append({
            "epoch": epoch,
            "phase": 2,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        print(
            f"  {epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
            f"{val_loss:>9.4f}  {val_acc:>7.2%}{flag}"
        )

    with open("models/transfer_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  Best MobileNetV2 validation accuracy: {best_acc:.2%}")
    print("  Saved checkpoint → models/mobilenet_best.pt")
    print("  Saved history    → models/transfer_history.json")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    model = MoodNetV2(num_classes=7)
    total, trainable = model.count_params()

    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")

    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)

    print(f"Input shape     : {list(dummy.shape)}")
    print(f"Output shape    : {list(out.shape)}")

    train_transfer(data_dir="data/")