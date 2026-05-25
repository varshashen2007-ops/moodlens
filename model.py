# ============================================================
# MoodLens — model.py
# The CNN architecture that classifies 7 emotions from faces
#
# Architecture: MoodNet (custom CNN + Transfer Learning ready)
#   Block 1: Edge detection (low-level features)
#   Block 2: Shape detection (eyes, nose, mouth curves)
#   Block 3: Complex pattern recognition (expressions)
#   Block 4: High-level semantic features
#   Head   : Fully connected classifier → 7 emotion scores
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    A more efficient convolution used in modern mobile architectures.
    Standard conv: one big operation.
    Depthwise separable: two smaller operations → same result, fewer params.
    Used in Google's MobileNet. We use it in the deeper blocks.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False          # one filter per channel
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False                # mix channels together
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.pointwise(self.depthwise(x))))


class ResidualBlock(nn.Module):
    """
    The key innovation from ResNet (2015) — adds a 'skip connection'.
    Instead of learning output = f(input), it learns output = f(input) + input.

    Why this matters: gradients flow directly back through the skip connection
    during training, so deep networks don't suffer from vanishing gradients.
    This lets us go deeper without the model getting worse.
    """
    def __init__(self, channels, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        residual = x                               # save the input
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + residual                       # ADD input back (the skip)
        return F.relu(out)


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation attention mechanism.
    Teaches the model to focus on the MOST IMPORTANT feature channels
    and suppress irrelevant ones — like a spotlight on what matters.

    Example: when detecting 'happy', the channels that captured
    mouth curves should be amplified; eye channels less so.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)         # Global Average Pool → 1x1
        self.fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()                           # output: 0–1 weight per channel
        )

    def forward(self, x):
        weights = self.fc(self.gap(x))             # shape: (batch, channels)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # → (batch, channels, 1, 1)
        return x * weights                         # rescale each channel


class MoodNet(nn.Module):
    """
    MoodLens custom CNN for 7-class facial emotion recognition.

    Input:  (batch, 1, 48, 48)  — grayscale 48x48 face crop
    Output: (batch, 7)          — raw scores (logits) for each emotion

    Architecture summary:
      Stem     → 1→32 channels,  48x48
      Block 1  → 32→64,          24x24  + ResidualBlock + Attention
      Block 2  → 64→128,         12x12  + ResidualBlock + Attention
      Block 3  → 128→256,        6x6    + ResidualBlock + Attention
      Block 4  → 256→512,        3x3
      Head     → Flatten → 512 → 256 → 7
    """

    def __init__(self, num_classes=7, dropout_rate=0.5):
        super().__init__()

        # ── Stem: first convolution, captures raw edges ──────────────
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ── Block 1: 32→64 channels, 48→24 spatial ──────────────────
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.res1  = ResidualBlock(64, dropout_rate=0.1)
        self.attn1 = ChannelAttention(64)

        # ── Block 2: 64→128 channels, 24→12 spatial ─────────────────
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),
        )
        self.res2  = ResidualBlock(128, dropout_rate=0.1)
        self.attn2 = ChannelAttention(128)

        # ── Block 3: 128→256 channels, 12→6 spatial ─────────────────
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, stride=2),
        )
        self.res3  = ResidualBlock(256, dropout_rate=0.15)
        self.attn3 = ChannelAttention(256)

        # ── Block 4: 256→512 channels, 6→3 spatial ──────────────────
        self.block4 = nn.Sequential(
            DepthwiseSeparableConv(256, 512, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ── Global pooling: collapses spatial dims → 1x1 ─────────────
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ── Classifier head ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes),
        )

        # ── Weight initialisation (important for stable training) ─────
        self._init_weights()

    def _init_weights(self):
        """He initialisation for conv layers, zeros for BN bias."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Block 1 → residual → attention
        x = self.block1(x)
        x = self.res1(x)
        x = self.attn1(x)

        # Block 2 → residual → attention
        x = self.block2(x)
        x = self.res2(x)
        x = self.attn2(x)

        # Block 3 → residual → attention
        x = self.block3(x)
        x = self.res3(x)
        x = self.attn3(x)

        # Block 4
        x = self.block4(x)

        # Pool + classify
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def predict_emotion(self, x):
        """
        Convenience method for inference (not training).
        Returns: (predicted_label_int, confidence_float, all_probs_tensor)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
        return pred.item(), conf.item(), probs.squeeze()


# ─────────────────────────────────────────────────────────────
# Model summary + sanity check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import EMOTION_LABELS

    print("\n" + "="*60)
    print("  MoodLens — MoodNet Architecture Check")
    print("="*60)

    model = MoodNet(num_classes=7, dropout_rate=0.5)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Total parameters    : {total_params:,}")
    print(f"  Trainable params    : {trainable:,}")
    print(f"  Model size (approx) : {total_params * 4 / 1024**2:.1f} MB")

    # Forward pass with a fake batch
    dummy = torch.randn(4, 1, 48, 48)   # batch=4, 1 channel, 48x48
    out   = model(dummy)
    print(f"\n  Input shape  : {list(dummy.shape)}")
    print(f"  Output shape : {list(out.shape)}  (4 images × 7 emotion scores)")

    # Test predict_emotion
    single = torch.randn(1, 1, 48, 48)
    pred, conf, probs = model.predict_emotion(single)
    print(f"\n  Sample prediction:")
    print(f"    Predicted : {EMOTION_LABELS[pred]} (label {pred})")
    print(f"    Confidence: {conf:.1%}")
    print(f"\n  All class probabilities:")
    for i, p in enumerate(probs):
        bar = "█" * int(p.item() * 40)
        print(f"    {EMOTION_LABELS[i]:10s}  {p.item():.3f}  {bar}")

    print("\n" + "="*60)
    print("  MoodNet ready. Next step: trainer.py")
    print("="*60 + "\n")
