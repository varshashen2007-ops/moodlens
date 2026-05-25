// frontend/src/pages/Session.jsx
// The live study session page.
// Shows: webcam feed preview, real-time emotion bars,
//        cognitive state badge, session timer, live chart.

import { useEffect, useRef, useState, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";

const API = "http://localhost:8000";

const EMOTION_COLORS = {
  happy: "#4ADE80", neutral: "#94A3B8", surprise: "#FBBF24",
  sad: "#818CF8", angry: "#F87171", fear: "#FB923C", disgust: "#C084FC",
};
const STATE_COLORS = {
  focused: "#4ADE80", engaged: "#FBBF24",
  fatigued: "#818CF8", stressed: "#F87171", distracted: "#C084FC",
};
const STATE_BG = {
  focused: "#041A0A", engaged: "#412402",
  fatigued: "#1A1535", stressed: "#1A0510", distracted: "#1A0520",
};

function EmotionBar({ label, value, color, dominant }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: dominant ? color : "#64748B", marginBottom: 4, fontWeight: dominant ? 500 : 400 }}>
        <span>{label}</span><span>{(value * 100).toFixed(0)}%</span>
      </div>
      <div style={{ height: 6, background: "#1E1E2E", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${value * 100}%`, background: dominant ? color : "#2D2D44", borderRadius: 3, transition: "width .3s ease" }} />
      </div>
    </div>
  );
}

export default function Session() {
  const videoRef      = useRef(null);
  const canvasRef     = useRef(null);
  const intervalRef   = useRef(null);
  const [sessionId,   setSessionId]   = useState(null);
  const [running,     setRunning]     = useState(false);
  const [elapsed,     setElapsed]     = useState(0);
  const [prediction,  setPrediction]  = useState(null);
  const [timeline,    setTimeline]    = useState([]);    // [{t, focus}]
  const [summary,     setSummary]     = useState(null);
  const [error,       setError]       = useState(null);

  // Timer
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setElapsed(e => e + 1), 1000);
    return () => clearInterval(id);
  }, [running]);

  const fmt = s => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  // Start webcam
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch {
      setError("Cannot access webcam. Allow camera permission in your browser.");
    }
  };

  // Stop webcam
  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      videoRef.current.srcObject = null;
    }
  };

  // Capture current frame → blob → send to /predict
  const captureAndPredict = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !sessionId) return;
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext("2d");
    canvas.width  = videoRef.current.videoWidth  || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    ctx.drawImage(videoRef.current, 0, 0);

    canvas.toBlob(async (blob) => {
      if (!blob) return;
      const form = new FormData();
      form.append("file", blob, "frame.jpg");

      try {
        const res  = await fetch(`${API}/predict`, { method: "POST", body: form });
        const pred = await res.json();
        setPrediction(pred);

        // Log to backend
        await fetch(`${API}/session/log`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            emotion: pred.emotion,
            confidence: pred.confidence,
            state: pred.state,
            all_probs: pred.all_probs,
          }),
        });

        // Live chart point
        const focusWeight = { focused: 100, engaged: 80, fatigued: 40, stressed: 20, distracted: 30 };
        setTimeline(t => [...t.slice(-30), { t: fmt(elapsed), focus: focusWeight[pred.state] || 50 }]);
      } catch { /* silent fail */ }
    }, "image/jpeg", 0.85);
  }, [sessionId, elapsed]);

  // Start session
  const startSession = async () => {
    setSummary(null);
    setTimeline([]);
    setElapsed(0);
    setPrediction(null);
    setError(null);

    try {
      const res  = await fetch(`${API}/session/start`, { method: "POST" });
      const data = await res.json();
      setSessionId(data.session_id);
      await startCamera();
      setRunning(true);
      intervalRef.current = setInterval(captureAndPredict, 800);
    } catch {
      setError("Could not start session. Is the backend running?");
    }
  };

  // End session
  const endSession = async () => {
    clearInterval(intervalRef.current);
    setRunning(false);
    stopCamera();

    try {
      const res  = await fetch(`${API}/session/end/${sessionId}`, { method: "POST" });
      const data = await res.json();
      setSummary(data);
    } catch {
      setError("Could not fetch session summary.");
    }
  };

  // Cleanup on unmount
  useEffect(() => () => { clearInterval(intervalRef.current); stopCamera(); }, []);

  // Keep capture interval in sync with sessionId
  useEffect(() => {
    if (running && sessionId) {
      clearInterval(intervalRef.current);
      intervalRef.current = setInterval(captureAndPredict, 800);
    }
  }, [running, sessionId, captureAndPredict]);

  const emotion = prediction?.emotion || "neutral";
  const state   = prediction?.state   || "neutral";
  const probs   = prediction?.all_probs || {};

  return (
    <div style={{ padding: 32 }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 600, color: "#E2E8F0", margin: 0 }}>Study Session</h1>
        <p style={{ fontSize: 13, color: "#64748B", marginTop: 4 }}>Your emotion is tracked every 0.8 seconds</p>
      </div>

      {error && (
        <div style={{ background: "#1A0510", border: "1px solid #4B1528", borderRadius: 10, padding: 16, color: "#F4C0D1", marginBottom: 20, fontSize: 13 }}>
          <i className="ti ti-alert-triangle" style={{ marginRight: 8 }} />{error}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 20, alignItems: "start" }}>
        {/* Left: Webcam + chart */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Webcam */}
          <div style={{ background: "#111119", border: "1px solid #1E1E2E", borderRadius: 12, overflow: "hidden", position: "relative" }}>
            <video ref={videoRef} autoPlay muted playsInline
              style={{ width: "100%", height: 340, objectFit: "cover", display: "block", background: "#0D0D14", transform: "scaleX(-1)" }} />
            <canvas ref={canvasRef} style={{ display: "none" }} />

            {/* Overlays */}
            {running && prediction && (
              <>
                <div style={{ position: "absolute", top: 16, left: 16, background: STATE_BG[state] || "#0D0D14", border: `1px solid ${STATE_COLORS[state] || "#7C6FFF"}`, borderRadius: 8, padding: "8px 14px" }}>
                  <div style={{ fontSize: 10, color: "#64748B", textTransform: "uppercase", letterSpacing: "0.06em" }}>Cognitive state</div>
                  <div style={{ fontSize: 16, fontWeight: 600, color: STATE_COLORS[state] || "#7C6FFF", marginTop: 2 }}>{state.toUpperCase()}</div>
                </div>
                <div style={{ position: "absolute", top: 16, right: 16, background: "rgba(13,13,20,0.85)", borderRadius: 8, padding: "8px 14px", textAlign: "right" }}>
                  <div style={{ fontSize: 10, color: "#64748B" }}>Session time</div>
                  <div style={{ fontSize: 20, fontWeight: 600, color: "#E2E8F0", fontFamily: "monospace" }}>{fmt(elapsed)}</div>
                </div>
                {!prediction.face_found && (
                  <div style={{ position: "absolute", bottom: 16, left: "50%", transform: "translateX(-50%)", background: "#1A0E03", border: "1px solid #854F0B", borderRadius: 8, padding: "6px 14px", fontSize: 12, color: "#FBBF24" }}>
                    No face detected — position yourself in frame
                  </div>
                )}
              </>
            )}

            {/* Start / Stop buttons */}
            <div style={{ position: "absolute", bottom: 16, right: 16, display: "flex", gap: 10 }}>
              {!running ? (
                <button onClick={startSession}
                  style={{ background: "#4ADE80", color: "#041A0A", border: "none", borderRadius: 8, padding: "10px 20px", fontSize: 13, fontWeight: 600, cursor: "pointer", display: "flex", alignItems: "center", gap: 6 }}>
                  <i className="ti ti-player-play" /> Start session
                </button>
              ) : (
                <button onClick={endSession}
                  style={{ background: "#F87171", color: "#1A0510", border: "none", borderRadius: 8, padding: "10px 20px", fontSize: 13, fontWeight: 600, cursor: "pointer", display: "flex", alignItems: "center", gap: 6 }}>
                  <i className="ti ti-player-stop" /> End session
                </button>
              )}
            </div>
          </div>

          {/* Live focus chart */}
          <div style={{ background: "#111119", border: "1px solid #1E1E2E", borderRadius: 12, padding: 20 }}>
            <div style={{ fontSize: 13, color: "#94A3B8", marginBottom: 14 }}>Live focus score</div>
            <ResponsiveContainer width="100%" height={130}>
              <LineChart data={timeline}>
                <XAxis dataKey="t" tick={{ fontSize: 10, fill: "#475569" }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#475569" }} axisLine={false} tickLine={false} width={28} />
                <Tooltip contentStyle={{ background: "#1E1E2E", border: "1px solid #2D2D44", borderRadius: 8, fontSize: 12 }} />
                <Line type="monotone" dataKey="focus" stroke="#7C6FFF" strokeWidth={2} dot={false} name="Focus" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Right: Emotion panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {/* Dominant emotion */}
          <div style={{ background: "#111119", border: "1px solid #1E1E2E", borderRadius: 12, padding: 20, textAlign: "center" }}>
            <div style={{ fontSize: 11, color: "#64748B", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 12 }}>Detected emotion</div>
            <div style={{ fontSize: 36, fontWeight: 700, color: EMOTION_COLORS[emotion] || "#7C6FFF", marginBottom: 6 }}>
              {emotion.toUpperCase()}
            </div>
            <div style={{ fontSize: 13, color: "#475569" }}>
              {prediction ? `${(prediction.confidence * 100).toFixed(0)}% confidence` : "Waiting for webcam..."}
            </div>
          </div>

          {/* Probability bars */}
          <div style={{ background: "#111119", border: "1px solid #1E1E2E", borderRadius: 12, padding: 20 }}>
            <div style={{ fontSize: 12, color: "#64748B", marginBottom: 14, textTransform: "uppercase", letterSpacing: "0.06em" }}>All emotions</div>
            {Object.entries(EMOTION_COLORS).map(([em, col]) => (
              <EmotionBar key={em} label={em} value={probs[em] || 0}
                color={col} dominant={em === emotion} />
            ))}
          </div>

          {/* Tips based on state */}
          <div style={{ background: STATE_BG[state] || "#111119", border: `1px solid ${STATE_COLORS[state] || "#1E1E2E"}`, borderRadius: 12, padding: 16 }}>
            <div style={{ fontSize: 11, color: STATE_COLORS[state] || "#7C6FFF", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
              {running ? "Tip" : "Ready to start"}
            </div>
            <div style={{ fontSize: 12.5, color: "#94A3B8", lineHeight: 1.6 }}>
              {!running ? "Start a session and sit in front of your webcam with good lighting." :
               state === "focused"    ? "Great focus! Keep it up. Consider a 5-min break every 25 minutes." :
               state === "stressed"   ? "You look stressed. Take a slow, deep breath. Roll your shoulders." :
               state === "fatigued"   ? "Fatigue detected. Look away from the screen for 20 seconds." :
               state === "distracted" ? "Mind wandering? Try the 2-minute rule: just start the next task." :
               state === "engaged"    ? "You look engaged! This is a great learning state." :
               "Monitoring your focus..."}
            </div>
          </div>
        </div>
      </div>

      {/* Session summary */}
      {summary && (
        <div style={{ marginTop: 28, background: "#111119", border: "1px solid #1E1E2E", borderRadius: 12, padding: 28 }}>
          <div style={{ fontSize: 16, fontWeight: 600, color: "#E2E8F0", marginBottom: 20 }}>
            Session complete — here's how it went
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16, marginBottom: 24 }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 11, color: "#64748B", marginBottom: 6 }}>Duration</div>
              <div style={{ fontSize: 24, fontWeight: 600, color: "#7C6FFF" }}>{fmt(summary.duration_seconds)}</div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 11, color: "#64748B", marginBottom: 6 }}>Focus score</div>
              <div style={{ fontSize: 24, fontWeight: 600, color: "#4ADE80" }}>{summary.focus_score}%</div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 11, color: "#64748B", marginBottom: 6 }}>Dominant emotion</div>
              <div style={{ fontSize: 18, fontWeight: 600, color: EMOTION_COLORS[summary.dominant_emotion] || "#7C6FFF" }}>
                {summary.dominant_emotion}
              </div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 11, color: "#64748B", marginBottom: 6 }}>Cognitive state</div>
              <div style={{ fontSize: 18, fontWeight: 600, color: STATE_COLORS[summary.dominant_state] || "#7C6FFF" }}>
                {summary.dominant_state}
              </div>
            </div>
          </div>
          <button onClick={() => { setSummary(null); setElapsed(0); }}
            style={{ background: "transparent", border: "1px solid #2D2D44", borderRadius: 8, padding: "8px 18px", color: "#94A3B8", cursor: "pointer", fontSize: 13 }}>
            Start new session
          </button>
        </div>
      )}
    </div>
  );
}
