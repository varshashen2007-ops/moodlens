# ================================================================
# MoodLens v2 — transfer_learning/mobilenet_model.py
#
# Replaces the custom CNN with a MobileNetV2 backbone pretrained
# on ImageNet. Fine-tuned for 7-class emotion recognition.
#
# Why MobileNetV2?
#   - Pretrained on 1.2M images → already knows edges, textures,
#     shapes. We only teach it emotion-specific features.
#   - Lightweight: runs real-time on CPU for webcam inference.
#   - Expected accuracy jump: 62% → 68-72% on FER-2013.
#
# Strategy: Two-phase training
#   Phase 1 (Feature extraction): freeze ALL MobileNet layers,
#     train only the new emotion head. Fast, stable.
#   Phase 2 (Fine-tuning): unfreeze last 3 MobileNet blocks,
#     train end-to-end with very low LR. Gets the gains.
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmotionHead(nn.Module):
    """
    The new classification head we attach on top of MobileNetV2.
    MobileNetV2 outputs a 1280-dim feature vector. We project that
    down to 7 emotion classes with dropout for regularisation.
    """
    def __init__(self, in_features: int = 1280, num_classes: int = 7,
                 dropout: float = 0.4):
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(x)


class MoodNetV2(nn.Module):
    """
    MobileNetV2 backbone + custom emotion head.

    Input:  (batch, 1, 48, 48)  — grayscale FER-2013 image
    Output: (batch, 7)          — emotion logits

    Note: MobileNetV2 expects 3-channel RGB. We convert grayscale
    to 3 channels by replacing the first conv layer's weight with
    a single-channel version (mean of the 3 original channels).
    This preserves the pretrained knowledge.
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.4):
        super().__init__()

        # Load MobileNetV2 pretrained on ImageNet
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # ── Adapt first conv for grayscale (1 channel → 3 channels trick) ──
        # Original first conv: weight shape (32, 3, 3, 3)
        # We average across the 3 input channels → (32, 1, 3, 3)
        orig_weight = backbone.features[0][0].weight.data  # (32, 3, 3, 3)
        new_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        new_conv.weight = nn.Parameter(orig_weight.mean(dim=1, keepdim=True))
        backbone.features[0][0] = new_conv

        # Keep all MobileNetV2 feature layers
        self.backbone = backbone.features  # 19 blocks

        # Global average pool (collapses spatial dims)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Our new emotion head (replaces the original 1000-class head)
        self.head = EmotionHead(
            in_features=1280,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Start in feature-extraction mode (backbone frozen)
        self.freeze_backbone()

    # ── Training phase controls ──────────────────────────────

    def freeze_backbone(self):
        """Phase 1: freeze all backbone layers. Only head trains."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  [MoodNetV2] Backbone FROZEN — training head only")

    def unfreeze_top_blocks(self, num_blocks: int = 3):
        """
        Phase 2: unfreeze the last N blocks for fine-tuning.
        MobileNetV2 has 19 feature blocks (indices 0-18).
        Unfreezing last 3 = blocks 16, 17, 18.
        """
        total = len(self.backbone)
        for i, block in enumerate(self.backbone):
            if i >= total - num_blocks:
                for param in block.parameters():
                    param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [MoodNetV2] Unfroze last {num_blocks} blocks — "
              f"trainable params: {trainable:,}")

    def unfreeze_all(self):
        """Unfreeze everything for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        print("  [MoodNetV2] All layers unfrozen")

    # ── Forward pass ─────────────────────────────────────────

    def forward(self, x):
        # x: (B, 1, 48, 48)
        features = self.backbone(x)        # (B, 1280, 2, 2)
        pooled   = self.pool(features)     # (B, 1280, 1, 1)
        flat     = pooled.flatten(1)       # (B, 1280)
        logits   = self.head(flat)         # (B, 7)
        return logits

    def predict(self, x):
        """Returns (label_int, confidence_float, all_probs_tensor)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        return pred.item(), conf.item(), probs.squeeze()

    def count_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ── Two-phase trainer ────────────────────────────────────────

def train_transfer(data_dir: str = "data/", device_str: str = "auto"):
    """
    Full two-phase transfer learning training loop.
    Call this from the project root: python transfer_learning/mobilenet_model.py
    """
    import os, sys, json, time
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data_loader import get_dataloaders, EMOTION_LABELS
    import torch.optim as optim

    # Device
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"\n{'='*60}")
    print("  MoodLens v2 — Transfer Learning (MobileNetV2)")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    train_loader, val_loader, class_weights = get_dataloaders(
        data_dir=data_dir, batch_size=64, num_workers=0
    )

    model     = MoodNetV2(num_classes=7, dropout=0.4).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=0.1
    )

    os.makedirs("models", exist_ok=True)
    history = []
    best_acc = 0.0

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss = correct = total = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                if train:
                    optimizer.zero_grad()
                out  = model(imgs)
                loss = criterion(out, labels)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                total_loss += loss.item() * imgs.size(0)
                correct    += (out.argmax(1) == labels).sum().item()
                total      += imgs.size(0)
        return total_loss / total, correct / total

    # ══ PHASE 1: Feature extraction — 15 epochs ═══════════
    print("── Phase 1: Feature extraction (backbone frozen) ──\n")
    model.freeze_backbone()
    total, trainable = model.count_params()
    print(f"  Total params: {total:,}  |  Trainable: {trainable:,}\n")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    print(f"  {'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*7}")

    for ep in range(1, 16):
        tl, ta = run_epoch(train_loader, train=True)
        vl, va = run_epoch(val_loader,   train=False)
        scheduler.step()
        flag = " ✓" if va > best_acc else ""
        if va > best_acc:
            best_acc = va
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "val_acc": va, "phase": 1},
                       "models/mobilenet_best.pt")
        print(f"  {ep:>3}  {tl:>8.4f}  {ta:>6.2%}  {vl:>8.4f}  {va:>6.2%}{flag}")
        history.append({"epoch": ep, "phase": 1, "train_acc": round(ta, 4),
                         "val_acc": round(va, 4)})

    # ══ PHASE 2: Fine-tuning — 20 epochs ══════════════════
    print(f"\n── Phase 2: Fine-tuning (last 3 blocks unfrozen) ──\n")
    model.unfreeze_top_blocks(num_blocks=3)

    # Lower LR for fine-tuning — don't destroy pretrained weights
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    print(f"\n  {'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}")
    print(f"  {'─'*3}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*7}")

    for ep in range(1, 21):
        tl, ta = run_epoch(train_loader, train=True)
        vl, va = run_epoch(val_loader,   train=False)
        scheduler.step()
        flag = " ✓ best" if va > best_acc else ""
        if va > best_acc:
            best_acc = va
            torch.save({"epoch": ep + 15, "model_state": model.state_dict(),
                        "val_acc": va, "phase": 2, "config": {"dropout": 0.4}},
                       "models/mobilenet_best.pt")
        print(f"  {ep+15:>3}  {tl:>8.4f}  {ta:>6.2%}  {vl:>8.4f}  {va:>6.2%}{flag}")
        history.append({"epoch": ep + 15, "phase": 2, "train_acc": round(ta, 4),
                         "val_acc": round(va, 4)})

    with open("models/transfer_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best val accuracy: {best_acc:.2%}")
    print(f"  Saved → models/mobilenet_best.pt\n")


if __name__ == "__main__":
    # Quick architecture check
    model = MoodNetV2(num_classes=7)
    total, trainable = model.count_params()
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}  (head only, backbone frozen)")

    dummy = torch.randn(4, 1, 48, 48)
    out   = model(dummy)
    print(f"Input  shape    : {list(dummy.shape)}")
    print(f"Output shape    : {list(out.shape)}")
    print("\nRun  python transfer_learning/mobilenet_model.py  to train.")

    # Uncomment to actually train:
    # train_transfer(data_dir="data/")
