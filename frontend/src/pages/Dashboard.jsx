// frontend/src/pages/Dashboard.jsx
// Analytics overview — focus trends, emotion distribution,
// session history stats. All data from /analytics endpoint.

import { useEffect, useState } from "react";
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip, ResponsiveContainer, Legend
} from "recharts";

const API = "http://localhost:8000";

const EMOTION_COLORS = {
  happy: "#4ADE80", neutral: "#94A3B8", surprise: "#FBBF24",
  sad: "#818CF8", angry: "#F87171", fear: "#FB923C", disgust: "#C084FC",
};
const STATE_COLORS = {
  focused: "#4ADE80", engaged: "#FBBF24",
  fatigued: "#818CF8", stressed: "#F87171", distracted: "#C084FC",
};

function StatCard({ label, value, sub, color = "#7C6FFF" }) {
  return (
    <div style={{ background: "#111119", border: "1px solid #1E1E2E", borderRadius: 12, padding: "20px 24px" }}>
      <div style={{ fontSize: 12, color: "#64748B", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 600, color, marginBottom: 4 }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: "#475569" }}>{sub}</div>}
    </div>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#1E1E2E", border: "1px solid #2D2D44", borderRadius: 8, padding: "10px 14px" }}>
      <div style={{ fontSize: 12, color: "#94A3B8", marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ fontSize: 13, color: p.color, fontWeight: 500 }}>{p.name}: {p.value}</div>
      ))}
    </div>
  );
};

export default function Dashboard() {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    fetch(`${API}/analytics`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => { setError("Cannot reach API. Start the backend first."); setLoading(false); });
  }, []);

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "60vh", color: "#64748B" }}>
      <i className="ti ti-loader" style={{ fontSize: 24, marginRight: 10 }} /> Loading analytics...
    </div>
  );

  if (error) return (
    <div style={{ padding: 40 }}>
      <div style={{ background: "#1A0510", border: "1px solid #4B1528", borderRadius: 12, padding: 24, color: "#F4C0D1" }}>
        <i className="ti ti-alert-triangle" style={{ marginRight: 8 }} />{error}
        <div style={{ marginTop: 10, fontSize: 12, color: "#94A3B8" }}>
          Run: <code style={{ background: "#0D0D14", padding: "2px 8px", borderRadius: 4 }}>uvicorn backend.main:app --reload --port 8000</code>
        </div>
      </div>
    </div>
  );

  if (data?.message) return (
    <div style={{ padding: 40 }}>
      <h2 style={{ color: "#E2E8F0", marginBottom: 8 }}>Welcome to MoodLens</h2>
      <p style={{ color: "#64748B" }}>No sessions yet. Go to <strong>Study Session</strong> to start tracking your focus.</p>
    </div>
  );

  const emotionPie = Object.entries(data.emotion_distribution || {}).map(([name, value]) => ({ name, value }));
  const statePie   = Object.entries(data.state_distribution   || {}).map(([name, value]) => ({ name, value }));
  const focusTrend = (data.daily_focus_trend || []).map(d => ({
    date: d.date.slice(5),        // "MM-DD"
    focus: d.avg_focus,
  }));

  return (
    <div style={{ padding: 32 }}>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 22, fontWeight: 600, color: "#E2E8F0", margin: 0 }}>Analytics Dashboard</h1>
        <p style={{ fontSize: 13, color: "#64748B", marginTop: 4 }}>Your study focus patterns at a glance</p>
      </div>

      {/* Stat cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16, marginBottom: 28 }}>
        <StatCard label="Total sessions"     value={data.total_sessions}            color="#7C6FFF" />
        <StatCard label="Study hours"        value={`${(data.total_study_minutes / 60).toFixed(1)}h`} color="#4ADE80" sub={`${data.total_study_minutes} minutes`} />
        <StatCard label="Avg focus score"    value={`${data.avg_focus_score}%`}     color="#FBBF24" sub="0 = distracted, 100 = focused" />
        <StatCard label="Best session focus" value={data.best_session ? `${data.best_session.focus}%` : "—"} color="#F472B6" sub={data.best_session?.date?.slice(0, 10)} />
      </div>

      {/* Charts row 1 */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 20, marginBottom: 20 }}>
        {/* Focus trend line */}
        <div style={{ background: "#111119", border: "1px solid #1E1E2E", borderRadius: 12, padding: 24 }}>
          <div style={{ fontSize: 14, fontWeight: 500, color: "#E2E8F0", marginBottom: 18 }}>Focus score trend</div>
          {focusTrend.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={focusTrend}>
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: "#64748B" }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: "#64748B" }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" dataKey="focus" stroke="#7C6FFF" strokeWidth={2} dot={{ fill: "#7C6FFF", r: 4 }} name="Focus %" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: 200, display: "flex", alignItems: "center", justifyContent: "center", color: "#475569", fontSize: 13 }}>
              Complete sessions to see your trend
            </div>
          )}
        </div>

        {/* State donut */}
        <div style={{ background: "#111119", border: "1px solid #1E1E2E", borderRadius: 12, padding: 24 }}>
          <div style={{ fontSize: 14, fontWeight: 500, color: "#E2E8F0", marginBottom: 18 }}>Cognitive states</div>
          <ResponsiveContainer width="100%" height={160}>
            <PieChart>
              <Pie data={statePie} cx="50%" cy="50%" innerRadius={45} outerRadius={70} paddingAngle={3} dataKey="value">
                {statePie.map((e, i) => <Cell key={i} fill={STATE_COLORS[e.name] || "#7C6FFF"} />)}
              </Pie>
              <Tooltip formatter={(v) => `${v}%`} contentStyle={{ background: "#1E1E2E", border: "1px solid #2D2D44", borderRadius: 8 }} />
            </PieChart>
          </ResponsiveContainer>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
            {statePie.map((e, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: "#94A3B8" }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: STATE_COLORS[e.name] || "#7C6FFF" }} />
                {e.name} {e.value}%
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Charts row 2 */}
      <div style={{ background: "#111119", border: "1px solid #1E1E2E", borderRadius: 12, padding: 24 }}>
        <div style={{ fontSize: 14, fontWeight: 500, color: "#E2E8F0", marginBottom: 18 }}>Emotion breakdown (all sessions)</div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={emotionPie} margin={{ left: -20 }}>
            <XAxis dataKey="name" tick={{ fontSize: 12, fill: "#64748B" }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 11, fill: "#64748B" }} axisLine={false} tickLine={false} unit="%" />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]} name="Percentage">
              {emotionPie.map((e, i) => <Cell key={i} fill={EMOTION_COLORS[e.name] || "#7C6FFF"} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
