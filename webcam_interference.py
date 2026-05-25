# ============================================================
# MoodLens — webcam_inference.py
# Live real-time emotion detection from your webcam.
# This is the "wow demo" file — runs MoodNet on each frame,
# overlays emotion bars, cognitive state, and session timer.
# ============================================================

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time
from collections import deque
from data_loader import EMOTION_LABELS, EMOTION_TO_STATE
from model import MoodNet

# ─── Load model ──────────────────────────────────────────────
CHECKPOINT = "models/best_model.pt"

def load_model():
    ckpt  = torch.load(CHECKPOINT, map_location="cpu")
    model = MoodNet(num_classes=7, dropout_rate=ckpt["config"]["dropout_rate"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

# ─── OpenCV face detector (built-in, no download needed) ─────
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ─── Colour scheme for each emotion ──────────────────────────
EMOTION_COLORS = {
    "happy"   : (74,  222, 128),   # green
    "neutral" : (148, 163, 184),   # slate
    "surprise": (251, 191,  36),   # yellow
    "sad"     : (99,  102, 241),   # indigo
    "angry"   : (248,  113, 113),  # red
    "fear"    : (217, 119,   6),   # amber
    "disgust" : (168,  85, 247),   # purple
}

STATE_COLORS = {
    "focused"    : (74,  222, 128),
    "engaged"    : (251, 191,  36),
    "fatigued"   : (99,  102, 241),
    "stressed"   : (248, 113, 113),
    "distracted" : (168,  85, 247),
}

def preprocess_face(face_img):
    """Crop → grayscale → resize → normalise → tensor."""
    gray   = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    tensor  = torch.tensor(resized, dtype=torch.float32) / 255.0
    tensor  = (tensor - 0.5) / 0.5               # normalise to [-1, 1]
    return tensor.unsqueeze(0).unsqueeze(0)       # → (1, 1, 48, 48)

def draw_rounded_rect(img, x1, y1, x2, y2, r, color, alpha=0.6):
    """Draw a semi-transparent rounded rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_emotion_bars(frame, probs, x, y, dominant_idx):
    """Draw probability bars for all 7 emotions."""
    bar_w, bar_h, gap = 120, 10, 18
    for i, (label, prob) in enumerate(zip(EMOTION_LABELS.values(), probs)):
        row_y   = y + i * gap
        color   = EMOTION_COLORS.get(label, (200, 200, 200))
        fill_w  = int(prob * bar_w)
        # background bar
        cv2.rectangle(frame, (x, row_y), (x + bar_w, row_y + bar_h),
                      (40, 40, 60), -1, cv2.LINE_AA)
        # fill bar
        if fill_w > 0:
            cv2.rectangle(frame, (x, row_y), (x + fill_w, row_y + bar_h),
                          color, -1, cv2.LINE_AA)
        # label text
        weight = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"{label[:7]:7s} {prob:.0%}",
                    (x + bar_w + 6, row_y + bar_h - 1),
                    weight, 0.32,
                    color if i == dominant_idx else (160, 160, 180),
                    1, cv2.LINE_AA)

def run_demo():
    if not torch.load.__module__:
        pass
    model = load_model()
    print("\n  MoodLens webcam inference running.")
    print("  Press  Q  to quit.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  Error: cannot open webcam. Make sure it's connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Smoothing: average predictions over last N frames
    prob_buffer = deque(maxlen=8)
    session_start = time.time()
    fps_times = deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps_times.append(time.time())
        fps = len(fps_times) / (fps_times[-1] - fps_times[0] + 1e-6) if len(fps_times) > 1 else 0

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        dominant_emotion = "neutral"
        dominant_idx     = 4
        smoothed_probs   = np.ones(7) / 7  # uniform if no face

        for (fx, fy, fw, fh) in faces[:1]:   # process only the largest face
            face_crop = frame[fy:fy+fh, fx:fx+fw]
            tensor    = preprocess_face(face_crop)

            with torch.no_grad():
                logits = model(tensor)
                probs  = F.softmax(logits, dim=1).squeeze().numpy()

            prob_buffer.append(probs)
            smoothed_probs = np.mean(prob_buffer, axis=0)

            dominant_idx     = int(np.argmax(smoothed_probs))
            dominant_emotion = EMOTION_LABELS[dominant_idx]
            face_color       = EMOTION_COLORS.get(dominant_emotion, (200, 200, 200))

            # Draw face box
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), face_color, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (fx, fy-28), (fx+fw, fy), face_color, -1)
            cv2.putText(frame, f"{dominant_emotion.upper()}  {smoothed_probs[dominant_idx]:.0%}",
                        (fx+6, fy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (10, 10, 20), 1, cv2.LINE_AA)

        # ── Left panel: emotion bars ──────────────────────────
        h, w = frame.shape[:2]
        draw_rounded_rect(frame, 10, 10, 280, 165, 8, (20, 20, 35), alpha=0.75)
        draw_emotion_bars(frame, smoothed_probs, 18, 20, dominant_idx)

        # ── Right panel: cognitive state ──────────────────────
        cog_state  = EMOTION_TO_STATE.get(dominant_emotion, "neutral")
        state_color = STATE_COLORS.get(cog_state, (200, 200, 200))

        draw_rounded_rect(frame, w-220, 10, w-10, 95, 8, (20, 20, 35), alpha=0.75)
        cv2.putText(frame, "COGNITIVE STATE",
                    (w-210, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 150), 1)
        cv2.putText(frame, cog_state.upper(),
                    (w-210, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    state_color, 2, cv2.LINE_AA)

        # ── Bottom bar: session timer + FPS ───────────────────
        elapsed  = int(time.time() - session_start)
        mins, secs = divmod(elapsed, 60)
        draw_rounded_rect(frame, 10, h-50, 300, h-10, 6, (20, 20, 35), alpha=0.7)
        cv2.putText(frame,
                    f"Session: {mins:02d}:{secs:02d}   FPS: {fps:.0f}",
                    (20, h-22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (160, 160, 180), 1, cv2.LINE_AA)

        # MoodLens watermark
        cv2.putText(frame, "MoodLens", (w-100, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 90), 1, cv2.LINE_AA)

        cv2.imshow("MoodLens — Live Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Session ended. Duration: {mins:02d}:{secs:02d}")


if __name__ == "__main__":
    import os
    if not os.path.exists(CHECKPOINT):
        print(f"\n  No model found at {CHECKPOINT}")
        print("  Train first with:  python trainer.py\n")
    else:
        run_demo()
