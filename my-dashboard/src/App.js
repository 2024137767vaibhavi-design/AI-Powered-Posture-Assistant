import { useEffect, useRef, useState } from "react";
import {
  doc,
  onSnapshot,
  setDoc,
  serverTimestamp,
  collection,
  addDoc,
} from "firebase/firestore";
import { db } from "./firebase/config";

import PostureCard from "./components/PostureCard";
import "./App.css";

function App() {
  // ---------- Firebase live posture state ----------
  const [posture, setPosture] = useState("correct");
  const [issue, setIssue] = useState("—");
  const [exercise, setExercise] = useState("Squat");

  const lastLoggedStatusRef = useRef("");

  // ---------- Backend control ----------
  const [backendRunning, setBackendRunning] = useState(false);
  const [backendMsg, setBackendMsg] = useState("");
  const API = "http://localhost:8000";

  const refreshBackendStatus = async () => {
    try {
      const res = await fetch(`${API}/status`);
      const data = await res.json();
      setBackendRunning(Boolean(data.running));
      setBackendMsg("");
    } catch {
      setBackendRunning(false);
      setBackendMsg("Backend server not reachable");
    }
  };

  const startBackend = async () => {
    setBackendMsg("Starting AI detection...");
    try {
      const res = await fetch(`${API}/start`, { method: "POST" });
      const data = await res.json();
      setBackendMsg(data.msg || "AI detection started");
      await refreshBackendStatus();
    } catch {
      setBackendMsg("Failed to start backend");
    }
  };

  const stopBackend = async () => {
    setBackendMsg("Stopping AI detection...");
    try {
      const res = await fetch(`${API}/stop`, { method: "POST" });
      const data = await res.json();
      setBackendMsg(data.msg || "AI detection stopped");
      await refreshBackendStatus();
    } catch {
      setBackendMsg("Failed to stop backend");
    }
  };

  // ---------- Firestore realtime listener ----------
  useEffect(() => {
    const ref = doc(db, "postureLogs", "latest");

    const unsubscribe = onSnapshot(ref, async (snapshot) => {
      if (!snapshot.exists()) {
        await setDoc(
          ref,
          {
            status: "correct",
            exercise: "Squat",
            issue: "—",
            timestamp: serverTimestamp(),
          },
          { merge: true }
        );
        return;
      }

      const data = snapshot.data() || {};

      const raw = String(data.status ?? data.posture ?? "")
        .trim()
        .toLowerCase();
      const normalized = raw === "correct" ? "correct" : "wrong";

      setPosture(normalized);
      setIssue(data.issue || "—");
      setExercise(data.exercise || "Squat");

      // Log WRONG posture events (once per transition)
      if (normalized === "wrong" && lastLoggedStatusRef.current !== "wrong") {
        await addDoc(collection(db, "postureHistory"), {
          status: "wrong",
          exercise: data.exercise || "Squat",
          issue: data.issue || "—",
          timestamp: serverTimestamp(),
        });
      }

      lastLoggedStatusRef.current = normalized;
    });

    return () => unsubscribe();
  }, []);

  // ---------- Check backend status on load ----------
  useEffect(() => {
    refreshBackendStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isCorrect = posture === "correct";

  return (
    <div className="page">
      {/* ---------- HEADER ---------- */}
      <div className="topbar">
        <div>
          <h1 className="title">🏋️ AI Gym Posture Assistant</h1>
          <p className="subtitle">
            Real-time posture detection & injury prevention
          </p>

          {/* Control buttons */}
          <div style={{ display: "flex", gap: 12, marginTop: 14 }}>
            <button
              onClick={startBackend}
              disabled={backendRunning}
              style={{
                padding: "10px 16px",
                borderRadius: 12,
                fontWeight: 800,
                cursor: backendRunning ? "not-allowed" : "pointer",
              }}
            >
              ▶ Start AI Detection
            </button>

            <button
              onClick={stopBackend}
              disabled={!backendRunning}
              style={{
                padding: "10px 16px",
                borderRadius: 12,
                fontWeight: 800,
                cursor: !backendRunning ? "not-allowed" : "pointer",
              }}
            >
              ⏹ Stop AI Detection
            </button>
          </div>

          {backendMsg && (
            <p style={{ color: "white", marginTop: 8 }}>{backendMsg}</p>
          )}
        </div>

        {/* Status badges */}
        <div className="badges">
          <span className="badge">Exercise: {exercise}</span>
          <span className={`badge ${isCorrect ? "ok" : "bad"}`}>
            {isCorrect ? "SAFE ✅" : "RISK ⚠️"}
          </span>
          <span className="badge">
            Backend: {backendRunning ? "RUNNING" : "OFF"}
          </span>
        </div>
      </div>

      {/* ---------- MAIN GRID ---------- */}
      <div className="grid">
        {/* Camera Stream Panel */}
        <div className="panel">
          <h3 style={{ color: "white", marginTop: 0 }}>
            📷 Live AI Camera (Skeleton)
          </h3>

          {backendRunning ? (
            <img
              src="http://localhost:8000/video"
              alt="AI Camera Stream"
              style={{
                width: "100%",
                maxWidth: 720,
                borderRadius: 16,
                border: "1px solid rgba(255,255,255,0.15)",
                boxShadow: "0 20px 40px rgba(0,0,0,0.35)",
              }}
            />
          ) : (
            <p style={{ color: "rgba(255,255,255,0.85)" }}>
              Click <b>Start AI Detection</b> to open the camera.
            </p>
          )}
        </div>

        {/* Posture Status Panel */}
        <div className="panel">
          <PostureCard status={posture} />

          <div className="miniCard">
            <h3 style={{ marginTop: 0 }}>🧾 Live Details</h3>

            <p>
              <b>Status:</b>{" "}
              <span style={{ color: isCorrect ? "#b7ffcf" : "#ffd0d0" }}>
                {isCorrect ? "Correct" : "Wrong"}
              </span>
            </p>

            <p>
              <b>Issue:</b> {issue}
            </p>

            {!isCorrect && (
              <div className="tips">
                <h4 style={{ margin: "10px 0 6px" }}>✅ Fix Tips</h4>
                <ul style={{ margin: 0, paddingLeft: "18px" }}>
                  <li>Keep spine neutral</li>
                  <li>Knees aligned with toes</li>
                  <li>Engage core muscles</li>
                  <li>Maintain controlled movement</li>
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
