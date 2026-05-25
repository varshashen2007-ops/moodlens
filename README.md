# MoodLens — Real-Time AI Emotion Recognition System

MoodLens is a deep learning–powered real-time emotion recognition system built using PyTorch, OpenCV, and computer vision techniques. The project detects facial emotions from live webcam input and maps them into cognitive study states such as focused, stressed, fatigued, and engaged.

This project was built as a complete end-to-end AI pipeline — from dataset preprocessing and CNN model training to evaluation, live inference, and deployment-ready architecture preparation.

---

# Features

- Real-time webcam emotion detection
- Custom CNN architecture built with PyTorch
- FER-2013 facial emotion dataset support
- Live probability visualization
- Cognitive state mapping system
- Evaluation dashboard with:
  - Confusion matrix
  - Accuracy curves
  - Loss curves
  - F1-score analysis
- Model checkpoint saving
- Training history logging
- Data augmentation pipeline
- Production-inspired CNN components

---

# Emotion Classes

The model predicts the following 7 emotions:

| Label | Emotion |
|---|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Neutral |
| 5 | Sad |
| 6 | Surprise |

---

# Cognitive State Mapping

MoodLens converts emotions into higher-level cognitive states:

| Emotion | Cognitive State |
|---|---|
| Happy | Focused |
| Neutral | Focused |
| Surprise | Engaged |
| Sad | Fatigued |
| Angry | Stressed |
| Fear | Stressed |
| Disgust | Distracted |

---

# Tech Stack

## AI / Deep Learning
- Python
- PyTorch
- TorchVision
- NumPy

## Computer Vision
- OpenCV
- MediaPipe

## Data Science & Evaluation
- Scikit-learn
- Matplotlib

## Frontend & Backend (Planned)
- FastAPI
- React.js

---

# Dataset

This project uses the FER-2013 facial emotion recognition dataset.

Dataset characteristics:
- 48x48 grayscale face images
- 7 emotion classes
- Real-world noisy facial expressions
- Standard benchmark dataset for emotion recognition

---

# Model Architecture — MoodNet

MoodNet is a custom CNN architecture inspired by modern production deep learning systems.

Key architectural features:
- Residual skip connections (ResNet-inspired)
- Depthwise separable convolutions (MobileNet-inspired)
- Squeeze-and-Excitation attention blocks (SENet-inspired)
- Batch normalization
- Dropout regularization
- Adaptive pooling

The network learns:
- Facial edge patterns
- Eye movement structures
- Mouth curvature
- Expression semantics
- Emotional micro-features

---

# Training Pipeline

The training system includes:

- AdamW optimizer
- Learning rate scheduling
- Label smoothing
- Gradient clipping
- Mixed precision training support
- Early stopping
- Automatic checkpoint saving
- Data augmentation

Training augmentations:
- Horizontal flipping
- Rotation
- Brightness variation
- Contrast variation

---

# Project Structure

```bash
moodlens/
│
├── data/
│   ├── train/
│   └── test/
│
├── models/
│   ├── best_model.pt
│   ├── evaluation_report.png
│   └── training_log.json
│
├── notebooks/
├── backend/
├── frontend/
│
├── data_loader.py
├── model.py
├── trainer.py
├── evaluate.py
├── webcam_interference.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/varshashen2007-ops/moodlens.git
cd moodlens
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

Place the FER-2013 dataset inside the `data/` folder:

```bash
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

---

# Running the Project

## Step 1 — Verify Data Loader

```bash
python data_loader.py
```

Expected output:
- Dataset statistics
- Batch shapes
- Class weights
- Successful loading confirmation

---

## Step 2 — Verify CNN Architecture

```bash
python model.py
```

Expected output:
- Model parameter count
- Output tensor shape
- Sample predictions

---

## Step 3 — Train the Model

```bash
python trainer.py
```

Training features:
- Saves best checkpoint automatically
- Logs training history
- Displays train/validation metrics each epoch

---

## Step 4 — Evaluate Model

```bash
python evaluate.py
```

Outputs:
- Classification report
- Confusion matrix
- F1 scores
- Evaluation dashboard image

Generated file:
```bash
models/evaluation_report.png
```

---

## Step 5 — Run Live Webcam Inference

```bash
python webcam_interference.py
```

Features:
- Live webcam feed
- Real-time emotion prediction
- Confidence visualization
- Cognitive state display
- Emotion smoothing for stable predictions

---

# Model Performance

Final validation accuracy achieved:

```text
62.39%
```

FER-2013 is a challenging benchmark dataset because:
- low-resolution images
- noisy labels
- difficult lighting conditions
- subtle facial expressions

This performance is consistent with many baseline research implementations.

---

# Evaluation Metrics

The project evaluates:
- Accuracy
- Precision
- Recall
- F1-score
- Per-class performance
- Confusion matrix analysis

---

# Sample Workflow

```text
Webcam Input
      ↓
Face Detection
      ↓
Image Preprocessing
      ↓
Tensor Conversion
      ↓
MoodNet CNN
      ↓
Emotion Prediction
      ↓
Cognitive State Mapping
      ↓
Live Display Overlay
```

---

# Future Improvements

Planned future upgrades:
- FastAPI backend deployment
- React dashboard frontend
- Session analytics
- Emotion tracking history
- Cloud deployment
- Mobile support
- Transformer-based emotion models
- Multi-face detection
- Audio emotion analysis

---

# Learning Outcomes

This project helped develop practical experience in:
- Deep learning
- CNN architecture design
- Computer vision
- Model evaluation
- PyTorch training workflows
- Real-time inference systems
- AI project structuring
- Git/GitHub version control

---

# Author

Varsha Shen

GitHub:
https://github.com/varshashen2007-ops

---

# License

This project is intended for educational and portfolio purposes.