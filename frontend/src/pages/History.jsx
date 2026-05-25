// frontend/src/pages/History.jsx
import { useEffect, useState } from "react";

const API = "http://localhost:8000";
const STATE_COLORS = {
  focused: "#4ADE80", engaged: "#FBBF24",
  fatigued: "#818CF8", stressed: "#F87171", distracted: "#C084FC",
};

export default function History() {
  const [sessions, setSessions] = useState([]);
  const [loading,  setLoading]  = useState(true);

  useEffect(() => {
    fetch(`${API}/history?limit=50`)
      .then(r => r.json())
      .then(d => { setSessions(d.sessions || []); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const fmt = s => {
    if (!s) return "--";
    const m = Math.floor(s / 60), sec = s % 60;
    return `${m}m ${sec}s`;
  };

  return (
    <div style={{ padding: 32 }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 600, color: "#E2E8F0", margin: 0 }}>Session history</h1>
        <p style={{ fontSize: 13, color: "#64748B", marginTop: 4 }}>{sessions.length} sessions recorded</p>
      </div>

      {loading ? (
        <div style={{ color: "#64748B" }}><i className="ti ti-loader" style={{ marginRight: 8 }} />Loading...</div>
      ) : sessions.length === 0 ? (
        <div style={{ color: "#64748B", fontSize: 14 }}>No completed sessions yet. Start your first session!</div>
      ) : (
        <div style={{ background: "#111119", border: "1px solid #1E1E2E", borderRadius: 12, overflow: "hidden" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #1E1E2E" }}>
                {["#", "Date", "Duration", "Dominant emotion", "State", "Focus score"].map(h => (
                  <th key={h} style={{ padding: "14px 18px", textAlign: "left", color: "#64748B", fontWeight: 500, fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sessions.map((s, i) => (
                <tr key={s.session_id} style={{ borderBottom: "1px solid #1A1A2E", transition: "background .1s" }}
                    onMouseEnter={e => e.currentTarget.style.background = "#15151F"}
                    onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
                  <td style={{ padding: "14px 18px", color: "#475569" }}>{s.session_id}</td>
                  <td style={{ padding: "14px 18px", color: "#94A3B8" }}>{new Date(s.started_at).toLocaleDateString()}</td>
                  <td style={{ padding: "14px 18px", color: "#E2E8F0", fontFamily: "monospace" }}>{fmt(s.duration_seconds)}</td>
                  <td style={{ padding: "14px 18px", color: "#94A3B8" }}>{s.dominant_emotion || "—"}</td>
                  <td style={{ padding: "14px 18px" }}>
                    <span style={{ background: `${STATE_COLORS[s.dominant_state] || "#7C6FFF"}22`, color: STATE_COLORS[s.dominant_state] || "#7C6FFF", padding: "3px 10px", borderRadius: 20, fontSize: 12, fontWeight: 500 }}>
                      {s.dominant_state || "—"}
                    </span>
                  </td>
                  <td style={{ padding: "14px 18px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <div style={{ flex: 1, height: 5, background: "#1E1E2E", borderRadius: 3, maxWidth: 80 }}>
                        <div style={{ height: "100%", width: `${s.focus_score || 0}%`, background: (s.focus_score || 0) > 60 ? "#4ADE80" : (s.focus_score || 0) > 35 ? "#FBBF24" : "#F87171", borderRadius: 3 }} />
                      </div>
                      <span style={{ color: "#E2E8F0", minWidth: 32 }}>{s.focus_score || 0}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
