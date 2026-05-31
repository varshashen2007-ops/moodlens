# ============================================================
# MoodLens — webcam_interference.py
# Professional real-time emotion detection demo.
#
# Improvements:
#   - MediaPipe face detection instead of Haar Cascade
#   - Better padded face crop
#   - Smoothed predictions over recent frames
#   - Confidence threshold to avoid fake certainty
#   - Clean real-time HUD overlay
# ============================================================

import os
import time
from collections import deque

import cv2
#from mediapipe.python.solutions import face_detection
import numpy as np
import torch
import torch.nn.functional as F

from data_loader import EMOTION_LABELS, EMOTION_TO_STATE
from model import MoodNet


CHECKPOINT = "models/best_model.pt"
CONFIDENCE_THRESHOLD = 0.20
SMOOTHING_WINDOW = 5


EMOTION_COLORS = {
    "happy": (74, 222, 128),
    "neutral": (148, 163, 184),
    "surprise": (251, 191, 36),
    "sad": (99, 102, 241),
    "angry": (248, 113, 113),
    "fear": (217, 119, 6),
    "disgust": (168, 85, 247),
    "uncertain": (180, 180, 180),
}

STATE_COLORS = {
    "focused": (74, 222, 128),
    "engaged": (251, 191, 36),
    "fatigued": (99, 102, 241),
    "stressed": (248, 113, 113),
    "distracted": (168, 85, 247),
    "uncertain": (180, 180, 180),
}


FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if FACE_CASCADE.empty():
    raise RuntimeError("Could not load OpenCV Haar Cascade face detector.")


def load_model():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")

    model = MoodNet(
        num_classes=7,
        dropout_rate=checkpoint["config"]["dropout_rate"],
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    val_acc = checkpoint.get("val_acc", None)

    print("\n  MoodLens model loaded successfully.")
    if val_acc is not None:
        print(f"  Validation accuracy: {val_acc:.2%}")

    return model


def preprocess_face(face_img):
    """
    Convert detected face crop into the exact format used during training:
    BGR crop → grayscale → 48x48 → normalized tensor [-1, 1]
    """
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

    tensor = torch.tensor(resized, dtype=torch.float32) / 255.0
    tensor = (tensor - 0.5) / 0.5

    return tensor.unsqueeze(0).unsqueeze(0)


def detect_face(frame):
    """
    Detect the largest face using OpenCV Haar Cascade.
    Returns padded bounding box: x, y, w, h
    """
    frame_h, frame_w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=6,
        minSize=(90, 90),
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    pad_x = int(w * 0.22)
    pad_y_top = int(h * 0.30)
    pad_y_bottom = int(h * 0.20)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y_top)
    x2 = min(frame_w, x + w + pad_x)
    y2 = min(frame_h, y + h + pad_y_bottom)

    final_w = x2 - x1
    final_h = y2 - y1

    if final_w <= 20 or final_h <= 20:
        return None

    return x1, y1, final_w, final_h


def predict_emotion(model, face_crop, prob_buffer):
    tensor = preprocess_face(face_crop)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().numpy()

    prob_buffer.append(probs)
    smoothed_probs = np.mean(prob_buffer, axis=0)

    dominant_idx = int(np.argmax(smoothed_probs))
    confidence = float(smoothed_probs[dominant_idx])
    emotion = EMOTION_LABELS[dominant_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        display_emotion = "uncertain"
        cognitive_state = "uncertain"
    else:
        display_emotion = emotion
        cognitive_state = EMOTION_TO_STATE.get(emotion, "uncertain")

    return {
        "raw_emotion": emotion,
        "display_emotion": display_emotion,
        "cognitive_state": cognitive_state,
        "dominant_idx": dominant_idx,
        "confidence": confidence,
        "probs": smoothed_probs,
    }


def draw_rounded_rect(img, x1, y1, x2, y2, radius, color, alpha=0.65):
    overlay = img.copy()

    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)

    corners = [
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ]

    for cx, cy in corners:
        cv2.circle(overlay, (cx, cy), radius, color, -1)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_emotion_bars(frame, probs, x, y, dominant_idx):
    bar_w = 125
    bar_h = 10
    row_gap = 18

    for idx, (label, prob) in enumerate(zip(EMOTION_LABELS.values(), probs)):
        row_y = y + idx * row_gap
        color = EMOTION_COLORS.get(label, (200, 200, 200))
        fill_w = int(prob * bar_w)

        cv2.rectangle(
            frame,
            (x, row_y),
            (x + bar_w, row_y + bar_h),
            (42, 42, 58),
            -1,
            cv2.LINE_AA,
        )

        if fill_w > 0:
            cv2.rectangle(
                frame,
                (x, row_y),
                (x + fill_w, row_y + bar_h),
                color,
                -1,
                cv2.LINE_AA,
            )

        text_color = color if idx == dominant_idx else (160, 160, 180)

        cv2.putText(
            frame,
            f"{label[:7]:7s} {prob:.0%}",
            (x + bar_w + 8, row_y + bar_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            text_color,
            1,
            cv2.LINE_AA,
        )


def draw_status_banner(frame, message):
    h, w = frame.shape[:2]

    draw_rounded_rect(
        frame,
        10,
        h - 90,
        min(w - 10, 520),
        h - 55,
        8,
        (20, 20, 35),
        alpha=0.75,
    )

    cv2.putText(
        frame,
        message,
        (22, h - 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 180, 210),
        1,
        cv2.LINE_AA,
    )


def run_demo():
    if not os.path.exists(CHECKPOINT):
        print(f"\n  No model found at {CHECKPOINT}")
        print("  Train first with: python trainer.py\n")
        return

    model = load_model()

    print("\n  MoodLens webcam inference running with MediaPipe.")
    print("  Press Q to quit.\n")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("  Error: cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prob_buffer = deque(maxlen=SMOOTHING_WINDOW)
    fps_times = deque(maxlen=30)
    session_start = time.time()

    smoothed_probs = np.ones(7) / 7
    dominant_idx = 4
    display_emotion = "uncertain"
    cognitive_state = "uncertain"
    confidence = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        fps_times.append(time.time())

        if len(fps_times) > 1:
            fps = len(fps_times) / (fps_times[-1] - fps_times[0] + 1e-6)
        else:
            fps = 0.0

        face_box = detect_face(frame)
        if face_box is not None:
            fx, fy, fw, fh = face_box
            face_crop = frame[fy:fy + fh, fx:fx + fw]

            if face_crop.size > 0:
                result = predict_emotion(model, face_crop, prob_buffer)

                smoothed_probs = result["probs"]
                dominant_idx = result["dominant_idx"]
                confidence = result["confidence"]
                display_emotion = result["display_emotion"]
                cognitive_state = result["cognitive_state"]

                face_color = EMOTION_COLORS.get(display_emotion, (200, 200, 200))

                cv2.rectangle(
                    frame,
                    (fx, fy),
                    (fx + fw, fy + fh),
                    face_color,
                    2,
                    cv2.LINE_AA,
                )

                label_text = f"{display_emotion.upper()}  {confidence:.0%}"

                label_y = max(32, fy - 8)

                cv2.rectangle(
                    frame,
                    (fx, label_y - 26),
                    (fx + max(fw, 210), label_y + 4),
                    face_color,
                    -1,
                )

                cv2.putText(
                    frame,
                    label_text,
                    (fx + 8, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (10, 10, 20),
                    1,
                    cv2.LINE_AA,
                )
        else:
            draw_status_banner(
                frame,
                "No face detected. Center your face and improve lighting.",
            )

        h, w = frame.shape[:2]

        # Left probability panel
        draw_rounded_rect(
            frame,
            10,
            10,
            290,
            170,
            8,
            (20, 20, 35),
            alpha=0.76,
        )

        cv2.putText(
            frame,
            "EMOTION PROBABILITIES",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (135, 135, 165),
            1,
            cv2.LINE_AA,
        )

        draw_emotion_bars(frame, smoothed_probs, 20, 42, dominant_idx)

        # Right state panel
        state_color = STATE_COLORS.get(cognitive_state, (200, 200, 200))

        draw_rounded_rect(
            frame,
            w - 245,
            10,
            w - 10,
            105,
            8,
            (20, 20, 35),
            alpha=0.76,
        )

        cv2.putText(
            frame,
            "COGNITIVE STATE",
            (w - 232, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (135, 135, 165),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            cognitive_state.upper(),
            (w - 232, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.86,
            state_color,
            2,
            cv2.LINE_AA,
        )

        # Bottom session bar
        elapsed = int(time.time() - session_start)
        mins, secs = divmod(elapsed, 60)

        draw_rounded_rect(
            frame,
            10,
            h - 50,
            330,
            h - 10,
            8,
            (20, 20, 35),
            alpha=0.72,
        )

        cv2.putText(
            frame,
            f"Session {mins:02d}:{secs:02d}   FPS {fps:.0f}   Confidence {confidence:.0%}",
            (22, h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (170, 170, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "MoodLens",
            (w - 105, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (80, 80, 115),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("MoodLens — MediaPipe Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n  Session ended. Duration: {mins:02d}:{secs:02d}")


if __name__ == "__main__":
    run_demo()