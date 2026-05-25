# ================================================================
# MoodLens v2 — backend/main.py
#
# Production FastAPI server. Exposes:
#   POST /predict          — single frame emotion prediction
#   POST /session/start    — start a study session
#   POST /session/end      — end session, return summary
#   POST /session/log      — log an emotion event mid-session
#   GET  /history          — fetch all past sessions
#   GET  /analytics        — aggregated emotion stats
#   GET  /health           — health check
#
# Run: uvicorn backend.main:app --reload --port 8000
# ================================================================

import io, os, sys, time, base64, json
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Text, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ── Path fix so imports work from backend/ subfolder ─────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_loader import EMOTION_LABELS, EMOTION_TO_STATE

# ── Database setup ────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "moodlens.db")
engine  = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
Base    = declarative_base()


class StudySession(Base):
    __tablename__ = "sessions"
    id          = Column(Integer, primary_key=True, index=True)
    started_at  = Column(DateTime, default=datetime.utcnow)
    ended_at    = Column(DateTime, nullable=True)
    duration_s  = Column(Integer, nullable=True)
    dominant_emotion   = Column(String, nullable=True)
    dominant_state     = Column(String, nullable=True)
    focus_score        = Column(Float, nullable=True)   # 0-100
    events      = relationship("EmotionEvent", back_populates="session",
                               cascade="all, delete-orphan")


class EmotionEvent(Base):
    __tablename__ = "emotion_events"
    id         = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    timestamp  = Column(Float, default=time.time)     # unix timestamp
    emotion    = Column(String)
    confidence = Column(Float)
    state      = Column(String)
    probs_json = Column(Text)                         # JSON array of 7 floats
    session    = relationship("StudySession", back_populates="events")


Base.metadata.create_all(engine)

# ── Model loading ─────────────────────────────────────────────
CHECKPOINT_PATHS = [
    "models/mobilenet_best.pt",   # transfer learning model (preferred)
    "models/best_model.pt",       # original MoodNet fallback
]

def load_model():
    """Load the best available checkpoint."""
    for path in CHECKPOINT_PATHS:
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu")
            # Detect which model class to use
            if "mobilenet" in path or ckpt.get("phase"):
                from transfer_learning.mobilenet_model import MoodNetV2
                cfg   = ckpt.get("config", {"dropout": 0.4})
                model = MoodNetV2(num_classes=7, dropout=cfg.get("dropout", 0.4))
            else:
                from model import MoodNet
                cfg   = ckpt.get("config", {"dropout_rate": 0.5})
                model = MoodNet(num_classes=7, dropout_rate=cfg.get("dropout_rate", 0.5))
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            print(f"  [API] Loaded model from {path}  "
                  f"(val_acc={ckpt.get('val_acc', '?'):.2%})")
            return model
    raise FileNotFoundError(
        "No model checkpoint found. Train with trainer.py first."
    )

# Load once at startup
MODEL = None
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="MoodLens API",
    description="Real-time emotion recognition and session analytics",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global MODEL
    MODEL = load_model()


# ── Pydantic schemas ──────────────────────────────────────────
class PredictResponse(BaseModel):
    emotion    : str
    confidence : float
    state      : str
    all_probs  : dict
    face_found : bool

class SessionStartResponse(BaseModel):
    session_id : int
    started_at : str

class LogEventRequest(BaseModel):
    session_id : int
    emotion    : str
    confidence : float
    state      : str
    all_probs  : dict

class SessionSummary(BaseModel):
    session_id       : int
    duration_seconds : int
    dominant_emotion : str
    dominant_state   : str
    focus_score      : float
    emotion_breakdown: dict
    timeline         : list

# ── Inference helpers ─────────────────────────────────────────
def preprocess_image(img_array: np.ndarray) -> torch.Tensor:
    """BGR array → grayscale → 48x48 → normalised tensor."""
    gray    = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    tensor  = torch.tensor(resized, dtype=torch.float32) / 255.0
    tensor  = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 48, 48)

def detect_face(img: np.ndarray):
    """Returns the largest face crop, or the full image if no face found."""
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return img, False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return img[y:y+h, x:x+w], True

def run_inference(img: np.ndarray):
    """Full pipeline: detect → crop → preprocess → infer."""
    face, found = detect_face(img)
    tensor      = preprocess_image(face)
    with torch.no_grad():
        logits = MODEL(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().numpy()
    pred_idx    = int(np.argmax(probs))
    emotion     = EMOTION_LABELS[pred_idx]
    confidence  = float(probs[pred_idx])
    state       = EMOTION_TO_STATE.get(emotion, "neutral")
    all_probs   = {EMOTION_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)}
    return emotion, confidence, state, all_probs, found

def compute_focus_score(events) -> float:
    """
    Compute a 0-100 focus score from a list of emotion events.
    Focused/engaged states add points; stressed/distracted subtract.
    """
    if not events:
        return 0.0
    weights = {"focused": 1.0, "engaged": 0.8, "fatigued": 0.2,
               "stressed": 0.0, "distracted": 0.1}
    total = sum(weights.get(e.state, 0.5) for e in events)
    return round(total / len(events) * 100, 1)

# ── Routes ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accept an image file (from webcam frame), return emotion prediction.
    Used by the React frontend every 500ms.
    """
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")

    contents = await file.read()
    arr      = np.frombuffer(contents, np.uint8)
    img      = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(400, "Invalid image")

    emotion, confidence, state, all_probs, face_found = run_inference(img)
    return PredictResponse(
        emotion=emotion, confidence=confidence,
        state=state, all_probs=all_probs, face_found=face_found
    )


@app.post("/session/start", response_model=SessionStartResponse)
def start_session():
    """Create a new study session in the database."""
    db   = Session()
    sess = StudySession(started_at=datetime.utcnow())
    db.add(sess)
    db.commit()
    db.refresh(sess)
    db.close()
    return SessionStartResponse(
        session_id=sess.id,
        started_at=sess.started_at.isoformat()
    )


@app.post("/session/log")
def log_event(req: LogEventRequest):
    """Log a single emotion detection event into an active session."""
    db    = Session()
    event = EmotionEvent(
        session_id=req.session_id,
        emotion=req.emotion,
        confidence=req.confidence,
        state=req.state,
        probs_json=json.dumps(req.all_probs),
        timestamp=time.time(),
    )
    db.add(event)
    db.commit()
    db.close()
    return {"logged": True}


@app.post("/session/end/{session_id}", response_model=SessionSummary)
def end_session(session_id: int):
    """
    Close a session, compute analytics, save to DB, return summary.
    """
    db   = Session()
    sess = db.query(StudySession).filter(StudySession.id == session_id).first()
    if not sess:
        db.close()
        raise HTTPException(404, f"Session {session_id} not found")

    events    = db.query(EmotionEvent)\
                  .filter(EmotionEvent.session_id == session_id)\
                  .order_by(EmotionEvent.timestamp)\
                  .all()
    ended_at  = datetime.utcnow()
    duration  = int((ended_at - sess.started_at).total_seconds())

    # Dominant emotion
    emotion_counts = {}
    for e in events:
        emotion_counts[e.emotion] = emotion_counts.get(e.emotion, 0) + 1
    dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
    dom_state = EMOTION_TO_STATE.get(dominant, "neutral")
    focus     = compute_focus_score(events)

    # Timeline (one entry per 10s bucket)
    timeline = []
    if events:
        t0 = events[0].timestamp
        bucket_size = 10
        buckets = {}
        for e in events:
            b = int((e.timestamp - t0) / bucket_size)
            if b not in buckets:
                buckets[b] = []
            buckets[b].append(e.state)
        for b in sorted(buckets):
            states = buckets[b]
            most_common = max(set(states), key=states.count)
            timeline.append({
                "second": b * bucket_size,
                "state": most_common,
                "count": len(states)
            })

    # Update DB
    sess.ended_at        = ended_at
    sess.duration_s      = duration
    sess.dominant_emotion = dominant
    sess.dominant_state  = dom_state
    sess.focus_score     = focus
    db.commit()
    db.close()

    breakdown = {k: round(v / len(events) * 100, 1) if events else 0
                 for k, v in emotion_counts.items()}

    return SessionSummary(
        session_id=session_id,
        duration_seconds=duration,
        dominant_emotion=dominant,
        dominant_state=dom_state,
        focus_score=focus,
        emotion_breakdown=breakdown,
        timeline=timeline,
    )


@app.get("/history")
def get_history(limit: int = 20):
    """Return the last N completed sessions."""
    db       = Session()
    sessions = db.query(StudySession)\
                 .filter(StudySession.ended_at != None)\
                 .order_by(StudySession.started_at.desc())\
                 .limit(limit).all()
    result = []
    for s in sessions:
        result.append({
            "session_id"      : s.id,
            "started_at"      : s.started_at.isoformat(),
            "duration_seconds": s.duration_s,
            "dominant_emotion": s.dominant_emotion,
            "dominant_state"  : s.dominant_state,
            "focus_score"     : s.focus_score,
        })
    db.close()
    return {"sessions": result, "total": len(result)}


@app.get("/analytics")
def get_analytics():
    """Aggregate stats across all sessions — for the dashboard charts."""
    db       = Session()
    sessions = db.query(StudySession)\
                 .filter(StudySession.ended_at != None).all()
    events   = db.query(EmotionEvent).all()

    if not sessions:
        db.close()
        return {"message": "No sessions yet"}

    # Avg focus score per day
    daily = {}
    for s in sessions:
        day = s.started_at.strftime("%Y-%m-%d")
        if day not in daily:
            daily[day] = []
        if s.focus_score:
            daily[day].append(s.focus_score)

    daily_avg = [{"date": d, "avg_focus": round(sum(v)/len(v), 1)}
                 for d, v in sorted(daily.items()) if v]

    # Emotion distribution across all events
    emotion_dist = {}
    state_dist   = {}
    for e in events:
        emotion_dist[e.emotion] = emotion_dist.get(e.emotion, 0) + 1
        state_dist[e.state]     = state_dist.get(e.state, 0) + 1

    total_events = len(events)
    emotion_pct  = {k: round(v / total_events * 100, 1) if total_events else 0
                    for k, v in emotion_dist.items()}
    state_pct    = {k: round(v / total_events * 100, 1) if total_events else 0
                    for k, v in state_dist.items()}

    # Best/worst sessions
    scored = [s for s in sessions if s.focus_score is not None]
    best   = max(scored, key=lambda s: s.focus_score) if scored else None
    worst  = min(scored, key=lambda s: s.focus_score) if scored else None

    db.close()
    return {
        "total_sessions"      : len(sessions),
        "total_study_minutes" : round(sum(s.duration_s or 0 for s in sessions) / 60, 1),
        "avg_focus_score"     : round(sum(s.focus_score or 0 for s in scored) / len(scored), 1) if scored else 0,
        "emotion_distribution": emotion_pct,
        "state_distribution"  : state_pct,
        "daily_focus_trend"   : daily_avg,
        "best_session"        : {"id": best.id, "focus": best.focus_score,
                                 "date": best.started_at.isoformat()} if best else None,
    }
