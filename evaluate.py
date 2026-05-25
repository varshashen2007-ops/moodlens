# ============================================================
# MoodLens — evaluate.py
# After training, this shows you:
#   - Confusion matrix (which emotions are confused with which)
#   - Per-class precision, recall, F1
#   - Training loss/accuracy curves
#   - Top mistakes the model makes
# ============================================================

import os, json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from data_loader import get_dataloaders, EMOTION_LABELS
from model import MoodNet


CHECKPOINT = "models/best_model.pt"
LOG_FILE   = "models/training_log.json"
OUTPUT_DIR = "models/"


def load_model(checkpoint_path):
    ckpt   = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["config"]
    model  = MoodNet(num_classes=7, dropout_rate=config["dropout_rate"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_acc = {ckpt['val_acc']:.2%})")
    return model, config


def get_all_predictions(model, val_loader):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_all(preds, labels, history):
    emotion_names = [EMOTION_LABELS[i] for i in range(7)]

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0F0F14")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    ACCENT  = "#7C6FFF"
    GREEN   = "#4ADE80"
    RED     = "#F87171"
    TEXT    = "#E2E8F0"
    SUBTEXT = "#94A3B8"
    BG      = "#1A1A2E"

    ax_style = dict(facecolor=BG)
    # ── 1. Confusion Matrix ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set(**ax_style)
    cm  = confusion_matrix(labels, preds, normalize="true")
    im  = ax1.imshow(cm, cmap="plasma", vmin=0, vmax=1)
    ax1.set_xticks(range(7)); ax1.set_yticks(range(7))
    ax1.set_xticklabels(emotion_names, rotation=45, ha="right", color=SUBTEXT, fontsize=8)
    ax1.set_yticklabels(emotion_names, color=SUBTEXT, fontsize=8)
    for i in range(7):
        for j in range(7):
            ax1.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                     color="white" if cm[i,j] > 0.4 else SUBTEXT, fontsize=7)
    plt.colorbar(im, ax=ax1, fraction=0.046)
    ax1.set_title("Confusion Matrix", color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Predicted", color=SUBTEXT, fontsize=9)
    ax1.set_ylabel("Actual", color=SUBTEXT, fontsize=9)
    ax1.tick_params(colors=SUBTEXT)

    # ── 2. Training curves ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set(**ax_style)
    epochs     = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"] for h in history]
    ax2.plot(epochs, train_loss, color=ACCENT, linewidth=2, label="Train Loss")
    ax2.plot(epochs, val_loss,   color=GREEN,  linewidth=2, label="Val Loss", linestyle="--")
    ax2.fill_between(epochs, train_loss, alpha=0.1, color=ACCENT)
    ax2.fill_between(epochs, val_loss,   alpha=0.1, color=GREEN)
    ax2.set_title("Loss Curves", color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", color=SUBTEXT, fontsize=9)
    ax2.set_ylabel("Loss", color=SUBTEXT, fontsize=9)
    ax2.tick_params(colors=SUBTEXT)
    ax2.legend(facecolor=BG, labelcolor=TEXT, fontsize=9)
    ax2.spines[:].set_color("#2D2D44")

    # ── 3. Accuracy curves ────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set(**ax_style)
    train_acc = [h["train_acc"] * 100 for h in history]
    val_acc   = [h["val_acc"]   * 100 for h in history]
    ax3.plot(epochs, train_acc, color=ACCENT, linewidth=2, label="Train Acc")
    ax3.plot(epochs, val_acc,   color=GREEN,  linewidth=2, label="Val Acc", linestyle="--")
    best_epoch = history[np.argmax([h["val_acc"] for h in history])]
    ax3.axvline(best_epoch["epoch"], color=RED, linewidth=1, linestyle=":", alpha=0.7)
    ax3.set_title("Accuracy Curves", color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax3.set_xlabel("Epoch", color=SUBTEXT, fontsize=9)
    ax3.set_ylabel("Accuracy (%)", color=SUBTEXT, fontsize=9)
    ax3.tick_params(colors=SUBTEXT)
    ax3.legend(facecolor=BG, labelcolor=TEXT, fontsize=9)
    ax3.spines[:].set_color("#2D2D44")

    # ── 4. Per-class F1 bar chart ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set(**ax_style)
    from sklearn.metrics import f1_score
    f1s    = f1_score(labels, preds, average=None)
    colors = [GREEN if f > 0.6 else ACCENT if f > 0.4 else RED for f in f1s]
    bars   = ax4.bar(emotion_names, f1s, color=colors, alpha=0.85, width=0.6)
    ax4.axhline(0.6, color=GREEN, linewidth=1, linestyle="--", alpha=0.5)
    ax4.set_ylim(0, 1)
    ax4.set_title("Per-class F1 Score", color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax4.set_xticklabels(emotion_names, rotation=30, ha="right", color=SUBTEXT, fontsize=8)
    ax4.tick_params(colors=SUBTEXT)
    ax4.spines[:].set_color("#2D2D44")
    for bar, f in zip(bars, f1s):
        ax4.text(bar.get_x() + bar.get_width()/2, f + 0.02,
                 f"{f:.2f}", ha="center", va="bottom", color=TEXT, fontsize=8)

    # ── 5. Precision vs Recall scatter ───────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set(**ax_style)
    from sklearn.metrics import precision_score, recall_score
    prec = precision_score(labels, preds, average=None, zero_division=0)
    rec  = recall_score(labels,  preds, average=None, zero_division=0)
    scatter_colors = plt.cm.plasma(np.linspace(0.2, 0.9, 7))
    for i, (p, r, name) in enumerate(zip(prec, rec, emotion_names)):
        ax5.scatter(r, p, s=100, color=scatter_colors[i], zorder=5)
        ax5.annotate(name, (r, p), xytext=(5, 5), textcoords="offset points",
                     color=TEXT, fontsize=8)
    ax5.plot([0, 1], [0, 1], color=SUBTEXT, linestyle="--", alpha=0.4)
    ax5.set_xlim(0, 1); ax5.set_ylim(0, 1)
    ax5.set_title("Precision vs Recall", color=TEXT, fontsize=11, fontweight="bold", pad=10)
    ax5.set_xlabel("Recall", color=SUBTEXT, fontsize=9)
    ax5.set_ylabel("Precision", color=SUBTEXT, fontsize=9)
    ax5.tick_params(colors=SUBTEXT)
    ax5.spines[:].set_color("#2D2D44")

    # ── 6. Overall stats summary ──────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set(**ax_style)
    ax6.axis("off")
    overall_acc = (preds == labels).mean()
    best_class  = emotion_names[np.argmax(f1s)]
    worst_class = emotion_names[np.argmin(f1s)]
    stats = [
        ("Overall Accuracy",  f"{overall_acc:.2%}"),
        ("Best Accuracy",     f"{best_class}  {np.max(f1s):.2f} F1"),
        ("Needs Work",        f"{worst_class}  {np.min(f1s):.2f} F1"),
        ("Mean F1 Score",     f"{f1s.mean():.3f}"),
        ("Total Val Samples", f"{len(labels):,}"),
        ("Training Epochs",   f"{len(history)}"),
    ]
    for i, (k, v) in enumerate(stats):
        y = 0.88 - i * 0.14
        ax6.text(0.05, y, k, color=SUBTEXT, fontsize=9, transform=ax6.transAxes)
        ax6.text(0.05, y - 0.06, v, color=GREEN if i < 2 else TEXT,
                 fontsize=11, fontweight="bold", transform=ax6.transAxes)

    ax6.set_title("Model Summary", color=TEXT, fontsize=11, fontweight="bold")
    fig.suptitle("MoodLens — MoodNet Evaluation Report", color=TEXT,
                 fontsize=14, fontweight="bold", y=0.98)

    out_path = os.path.join(OUTPUT_DIR, "evaluation_report.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Report saved → {out_path}")
    return out_path


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MoodLens — Model Evaluation")
    print("="*60)

    if not os.path.exists(CHECKPOINT):
        print(f"\n  No checkpoint found at {CHECKPOINT}")
        print("  Run  python trainer.py  first to train the model.\n")
        exit(1)

    model, config = load_model(CHECKPOINT)

    _, val_loader, _ = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=0,
    )

    print("  Running inference on validation set...")
    preds, labels, probs = get_all_predictions(model, val_loader)

    print("\n  Per-class report:\n")
    emotion_names = [EMOTION_LABELS[i] for i in range(7)]
    print(classification_report(labels, preds, target_names=emotion_names, digits=3))

    with open(LOG_FILE) as f:
        log  = json.load(f)
        history = log["history"]

    print("  Generating evaluation plots...")
    plot_all(preds, labels, history)

    print("\n  Done! Open models/evaluation_report.png to see your results.")
    print("  Next: run  python webcam_inference.py  for live emotion detection.\n")
