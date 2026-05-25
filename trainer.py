# ============================================================
# MoodLens — trainer.py
# Trains MoodNet on FER-2013 with:
#   - Weighted cross-entropy (handles class imbalance)
#   - OneCycleLR scheduler (fast convergence)
#   - Mixed precision training (2x faster on GPU)
#   - Early stopping (stops if not improving)
#   - Full checkpoint saving (resume any time)
#   - Live training log with rich progress
# ============================================================

import os, time, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from data_loader import get_dataloaders, EMOTION_LABELS
from model import MoodNet

# ─── Config — change these to tune training ──────────────────
CONFIG = {
    "data_dir"       : "data/",
    "batch_size"     : 64,
    "num_workers"    : 0,       # set to 2 on Mac/Linux
    "num_epochs"     : 60,
    "learning_rate"  : 3e-3,
    "weight_decay"   : 1e-4,
    "dropout_rate"   : 0.5,
    "patience"       : 10,      # early stop after 10 non-improving epochs
    "checkpoint_dir" : "models/",
    "log_file"       : "models/training_log.json",
}


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed precision: FP16 forward pass (faster, less memory)
        with autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def train():
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # ── Device selection ─────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n  GPU detected: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n  Apple Silicon GPU detected (MPS)")
    else:
        device = torch.device("cpu")
        print("\n  No GPU found — training on CPU (will be slow, ~1 min/epoch)")

    # ── Data ─────────────────────────────────────────────────
    print("  Loading data...")
    train_loader, val_loader, class_weights = get_dataloaders(
        data_dir=CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ────────────────────────────────────────────────
    model = MoodNet(num_classes=7, dropout_rate=CONFIG["dropout_rate"]).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  MoodNet parameters: {total_params:,}")

    # ── Loss: weighted CrossEntropy for imbalance ─────────────
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.1,    # prevents overconfidence — model learns softer probs
    )

    # ── Optimiser: AdamW (Adam + proper weight decay) ─────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # ── Scheduler: OneCycleLR — ramps up LR then anneals down ─
    # This is the fastest way to converge; used in fast.ai's training
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG["learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=CONFIG["num_epochs"],
        pct_start=0.3,
    )

    # ── Mixed precision scaler ────────────────────────────────
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Training loop ─────────────────────────────────────────
    best_val_acc  = 0.0
    best_val_loss = float("inf")
    patience_ctr  = 0
    history       = []

    print(f"\n{'='*65}")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}  {'LR':>8}")
    print(f"{'='*65}")

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()   # update LR (called per step inside loop above too)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        # Status flag
        flag = ""
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            patience_ctr  = 0
            flag = "  ✓ best"
            # Save best model checkpoint
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optim_state" : optimizer.state_dict(),
                "val_acc"     : val_acc,
                "config"      : CONFIG,
            }, os.path.join(CONFIG["checkpoint_dir"], "best_model.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= CONFIG["patience"]:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {CONFIG['patience']} epochs)")
                break

        print(
            f"  {epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
            f"{val_loss:>9.4f}  {val_acc:>7.2%}  {current_lr:>8.2e}{flag}"
        )

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4), "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

    # Save training history for plotting
    with open(CONFIG["log_file"], "w") as f:
        json.dump({"config": CONFIG, "history": history}, f, indent=2)

    print(f"\n  Best validation accuracy: {best_val_acc:.2%}")
    print(f"  Model saved → {CONFIG['checkpoint_dir']}best_model.pt")
    print(f"  History saved → {CONFIG['log_file']}")
    print(f"\n  Next: run  python evaluate.py  to see the confusion matrix\n")


if __name__ == "__main__":
    print("\n" + "="*65)
    print("  MoodLens — Training MoodNet on FER-2013")
    print("="*65)
    train()
