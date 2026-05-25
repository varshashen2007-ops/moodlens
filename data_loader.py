# ============================================================
# MoodLens — data_loader.py
# What this file does:
#   1. Finds the FER-2013 images on your computer
#   2. Converts them into numbers (pixels) PyTorch can read
#   3. Applies augmentations to make the model more robust
#   4. Packages everything into DataLoaders (batched feeds)
# ============================================================

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ─────────────────────────────────────────────
# STEP 1: Define what to do to every image
# ─────────────────────────────────────────────
# FER-2013 images are 48x48 pixels, grayscale (1 colour channel).
# We need to:
#   - Resize to 48x48 (already correct, but good to be explicit)
#   - Convert to a PyTorch Tensor (array of numbers)
#   - Normalise: shift pixel values from [0, 255] to [-1, 1]
#     so the model learns faster and more stably

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ensure single channel
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(p=0.5),       # flip 50% of images → model sees more variety
    transforms.RandomRotation(degrees=10),         # tilt up to 10° → handles tilted heads
    transforms.ColorJitter(brightness=0.2,         # slight brightness variation
                           contrast=0.2),          # slight contrast variation
    transforms.ToTensor(),                         # converts PIL image to [C, H, W] tensor
    transforms.Normalize(mean=[0.5],               # normalise: (pixel - 0.5) / 0.5
                         std=[0.5]),               # output range: [-1, 1]
])

# Validation/Test images: NO augmentation.
# We want to measure real-world performance, not augmented images.
VAL_TRANSFORMS = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# ─────────────────────────────────────────────
# STEP 2: The 7 emotion classes in FER-2013
# ─────────────────────────────────────────────
# These match the folder names inside data/train/ exactly.
EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

# Reverse lookup: "happy" → 3
LABEL_TO_IDX = {v: k for k, v in EMOTION_LABELS.items()}

# For the study-state logic, we map emotions to cognitive states
EMOTION_TO_STATE = {
    "happy":    "focused",
    "neutral":  "focused",
    "surprise": "engaged",
    "sad":      "fatigued",
    "angry":    "stressed",
    "fear":     "stressed",
    "disgust":  "distracted",
}


# ─────────────────────────────────────────────
# STEP 3: Build the DataLoaders
# ─────────────────────────────────────────────
def get_dataloaders(data_dir: str, batch_size: int = 64, num_workers: int = 2):
    """
    Reads FER-2013 from disk and returns train + validation DataLoaders.

    Args:
        data_dir    : path to the folder that contains 'train/' and 'test/'
                      e.g. "data/"  (relative) or "/home/user/moodlens/data/"
        batch_size  : how many images to feed the model at once (default 64)
                      Lower this to 32 if you run out of RAM.
        num_workers : parallel workers for loading images (set to 0 on Windows
                      if you get a BrokenPipeError)

    Returns:
        train_loader  : DataLoader for training
        val_loader    : DataLoader for validation
        class_weights : tensor to handle class imbalance (disgust has very few samples)
    """

    train_path = os.path.join(data_dir, "train")
    test_path  = os.path.join(data_dir, "test")

    # Sanity check — tell user exactly what's wrong if the folder is missing
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"\n[MoodLens] Could not find: {train_path}\n"
            f"Make sure you unzipped fer2013.zip into the data/ folder.\n"
            f"Expected structure:\n"
            f"  data/\n"
            f"    train/\n"
            f"      angry/\n"
            f"      happy/\n"
            f"      ... (7 folders)\n"
            f"    test/\n"
            f"      angry/\n"
            f"      ... (7 folders)\n"
        )

    # ImageFolder automatically assigns labels based on subfolder names
    # data/train/happy/img001.jpg  →  label = index of "happy"
    train_dataset = datasets.ImageFolder(root=train_path, transform=TRAIN_TRANSFORMS)
    val_dataset   = datasets.ImageFolder(root=test_path,  transform=VAL_TRANSFORMS)

    # ── Class weight calculation ──────────────────────────────────────
    # FER-2013 is IMBALANCED: "disgust" has ~500 images, "happy" has ~8,000.
    # Without correction, the model just learns to predict "happy" all the time.
    # We weight each class inversely to how common it is.
    class_counts = torch.zeros(len(EMOTION_LABELS))
    for _, label in train_dataset.samples:
        class_counts[label] += 1

    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()  # normalise to sum = 1

    # ── DataLoaders ───────────────────────────────────────────────────
    # shuffle=True for training → model sees images in random order each epoch
    # shuffle=False for validation → consistent results
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,   # faster GPU transfer if you have a GPU
        drop_last=True,    # drop the last incomplete batch (avoids BatchNorm issues)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, class_weights


# ─────────────────────────────────────────────
# STEP 4: Quick sanity check (run this file directly)
# ─────────────────────────────────────────────
# When you run:   python data_loader.py
# It will print a report so you know the data loaded correctly.

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  MoodLens — Data Loader Sanity Check")
    print("="*55)

    try:
        train_loader, val_loader, class_weights = get_dataloaders(
            data_dir="data/",
            batch_size=64,
            num_workers=0,   # 0 for Windows; 2+ for Mac/Linux
        )

        # Peek at one batch
        images, labels = next(iter(train_loader))

        print(f"\n✓ Data loaded successfully!")
        print(f"\n  Training batches  : {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"\n  Batch shape  : {images.shape}")
        print(f"               → (batch_size=64, channels=1, height=48, width=48)")
        print(f"\n  Pixel range  : {images.min():.2f}  to  {images.max():.2f}")
        print(f"               → should be close to -1.0 and +1.0")
        print(f"\n  Class weights (higher = rarer class):")
        for idx, weight in enumerate(class_weights):
            label = EMOTION_LABELS[idx]
            bar = "█" * int(weight * 200)
            print(f"    {label:10s}  {weight:.4f}  {bar}")

        print(f"\n  Label names from folder scan:")
        import torchvision
        ds = torchvision.datasets.ImageFolder("data/train")
        for idx, name in enumerate(ds.classes):
            print(f"    {idx} → {name}")

        print("\n" + "="*55)
        print("  Everything looks good. Ready to build the CNN.")
        print("="*55 + "\n")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"\n[Error] {e}")
        print("Check that your data/ folder has the correct structure.\n")