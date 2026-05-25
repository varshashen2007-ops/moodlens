// frontend/src/App.jsx
// Root of the MoodLens React app.
// Routes: /         → Dashboard (analytics overview)
//         /session  → Live study session with webcam
//         /history  → Past sessions table

import { useState, useEffect } from "react";
import Dashboard from "./pages/Dashboard";
import Session from "./pages/Session";
import History from "./pages/History";

const NAV = [
  { id: "dashboard", label: "Dashboard",    icon: "ti-layout-dashboard" },
  { id: "session",   label: "Study Session", icon: "ti-brain"            },
  { id: "history",   label: "History",       icon: "ti-history"          },
];

export default function App() {
  const [page, setPage]     = useState("dashboard");
  const [online, setOnline] = useState(false);

  // Poll backend health every 5s
  useEffect(() => {
    const check = async () => {
      try {
        const r = await fetch("http://localhost:8000/health");
        setOnline(r.ok);
      } catch {
        setOnline(false);
      }
    };
    check();
    const id = setInterval(check, 5000);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={{ display: "flex", minHeight: "100vh", fontFamily: "'Inter', system-ui, sans-serif", background: "#0D0D14", color: "#E2E8F0" }}>
      {/* Sidebar */}
      <aside style={{ width: 220, background: "#111119", borderRight: "1px solid #1E1E2E", display: "flex", flexDirection: "column", padding: "24px 0" }}>
        {/* Logo */}
        <div style={{ padding: "0 20px 28px", borderBottom: "1px solid #1E1E2E" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 32, height: 32, borderRadius: 8, background: "linear-gradient(135deg, #7C6FFF, #4ADE80)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <i className="ti ti-eye" style={{ color: "#fff", fontSize: 17 }} />
            </div>
            <div>
              <div style={{ fontWeight: 600, fontSize: 15, color: "#E2E8F0" }}>MoodLens</div>
              <div style={{ fontSize: 11, color: "#64748B" }}>v2.0 — AI Study Assistant</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav style={{ padding: "16px 12px", flex: 1 }}>
          {NAV.map(n => (
            <button key={n.id} onClick={() => setPage(n.id)}
              style={{
                display: "flex", alignItems: "center", gap: 10,
                width: "100%", padding: "10px 12px", borderRadius: 8,
                border: "none", cursor: "pointer", marginBottom: 4,
                background: page === n.id ? "#1E1E35" : "transparent",
                color: page === n.id ? "#7C6FFF" : "#94A3B8",
                fontSize: 13.5, fontWeight: page === n.id ? 500 : 400,
                transition: "all .15s",
              }}>
              <i className={`ti ${n.icon}`} style={{ fontSize: 17 }} />
              {n.label}
            </button>
          ))}
        </nav>

        {/* API status */}
        <div style={{ padding: "16px 20px", borderTop: "1px solid #1E1E2E" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: online ? "#4ADE80" : "#F87171" }} />
            <span style={{ color: "#64748B" }}>API {online ? "connected" : "offline"}</span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main style={{ flex: 1, overflow: "auto" }}>
        {page === "dashboard" && <Dashboard />}
        {page === "session"   && <Session   />}
        {page === "history"   && <History   />}
      </main>
    </div>
  );
}
