# control_server.py
# FastAPI backend that:
# 1) Starts/stops AI detection from frontend buttons
# 2) Opens webcam ONCE in backend
# 3) Runs MediaPipe Pose + draws skeleton on frames
# 4) Auto-detects exercise: Squat / Deadlift / Bicep Curl / Shoulder Press
# 5) Checks form -> status: correct/wrong + issue
# 6) Streams video to frontend at /video (MJPEG)
# 7) Writes live state to Firestore: postureLogs/latest
# 8) Logs wrong events to postureHistory

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
import math

import firebase_admin
from firebase_admin import credentials, firestore

# -------------------- Firebase --------------------
# Put serviceAccountKey.json in the SAME folder as this file.
cred = credentials.Certificate("serviceAccountKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# -------------------- FastAPI --------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- MediaPipe --------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# -------------------- Globals --------------------
running = False
cap = None
latest_jpeg = None
lock = threading.Lock()
worker_thread = None


# -------------------- Geometry helpers --------------------
def angle3(a, b, c):
    """Angle ABC in degrees using numpy vectors for (x,y)."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def line_angle_deg(p1, p2):
    """
    Absolute angle of the line p1->p2 relative to horizontal (0..90 approx).
    Bigger value => more vertical-ish/steeper.
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(math.degrees(math.atan2((y2 - y1), (x2 - x1))))


# -------------------- Exercise classification --------------------
def classify_exercise(lm):
    """
    Rule-based exercise guess (front camera works best).
    Returns one of: "Squat", "Deadlift", "Bicep Curl", "Shoulder Press"
    """
    # Basic points
    l_sh = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
    r_sh = (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
    l_el = (lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y)
    r_el = (lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y)
    l_wr = (lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y)
    r_wr = (lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y)

    l_hp = (lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y)
    r_hp = (lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y)
    l_kn = (lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y)
    r_kn = (lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y)
    l_an = (lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y)
    r_an = (lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y)

    # Angles
    l_elbow = angle3(l_sh, l_el, l_wr)
    r_elbow = angle3(r_sh, r_el, r_wr)
    l_knee = angle3(l_hp, l_kn, l_an)
    r_knee = angle3(r_hp, r_kn, r_an)

    # Shoulder press: wrists clearly above shoulders (remember: smaller y = higher)
    wrists_above_shoulders = (l_wr[1] < l_sh[1] - 0.03) or (r_wr[1] < r_sh[1] - 0.03)
    if wrists_above_shoulders:
        return "Shoulder Press"

    # Bicep curl: elbow flexed (angle smaller) and wrists not above shoulders
    elbow_flexed = (l_elbow < 110) or (r_elbow < 110)
    wrists_not_overhead = (l_wr[1] > l_sh[1] - 0.01) and (r_wr[1] > r_sh[1] - 0.01)
    if elbow_flexed and wrists_not_overhead:
        return "Bicep Curl"

    # Deadlift vs squat:
    # - Deadlift: knees more straight + torso hinge forward
    knees_straightish = (l_knee > 140) and (r_knee > 140)
    mid_sh = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    mid_hp = ((l_hp[0] + r_hp[0]) / 2, (l_hp[1] + r_hp[1]) / 2)
    torso_lean = line_angle_deg(mid_sh, mid_hp)  # bigger => more leaning

    if knees_straightish and torso_lean > 25:
        return "Deadlift"

    return "Squat"


# -------------------- Form checks --------------------
def check_form(exercise, lm):
    """
    Returns (status, issue).
    status: "correct" or "wrong"
    issue: string message
    """
    l_sh = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
    r_sh = (lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
    l_hp = (lm[mp_pose.PoseLandmark.LEFT_HIP].x, lm[mp_pose.PoseLandmark.LEFT_HIP].y)
    r_hp = (lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y)

    l_el = (lm[mp_pose.PoseLandmark.LEFT_ELBOW].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y)
    r_el = (lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y)
    l_wr = (lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y)
    r_wr = (lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y)

    l_kn = (lm[mp_pose.PoseLandmark.LEFT_KNEE].x, lm[mp_pose.PoseLandmark.LEFT_KNEE].y)
    r_kn = (lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y)
    l_an = (lm[mp_pose.PoseLandmark.LEFT_ANKLE].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y)
    r_an = (lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y)

    # Angles
    l_knee = angle3(l_hp, l_kn, l_an)
    r_knee = angle3(r_hp, r_kn, r_an)
    l_elbow = angle3(l_sh, l_el, l_wr)
    r_elbow = angle3(r_sh, r_el, r_wr)

    mid_sh = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    mid_hp = ((l_hp[0] + r_hp[0]) / 2, (l_hp[1] + r_hp[1]) / 2)
    torso_lean = line_angle_deg(mid_sh, mid_hp)

    status = "correct"
    issue = "—"

    if exercise == "Squat":
        if min(l_knee, r_knee) > 165:
            status, issue = "wrong", "Not squatting (legs too straight)"
        elif min(l_knee, r_knee) < 65:
            status, issue = "wrong", "Too deep / knee overbend"
        elif torso_lean > 55:
            status, issue = "wrong", "Back leaning too much"

    elif exercise == "Deadlift":
        if min(l_knee, r_knee) < 120:
            status, issue = "wrong", "Knees bending too much (looks like squat)"
        elif torso_lean < 20:
            status, issue = "wrong", "Not hinging (too upright)"
        elif torso_lean > 70:
            status, issue = "wrong", "Back angle too aggressive (risk)"

    elif exercise == "Bicep Curl":
        # cheating if elbow rises near shoulder height (y smaller = higher)
        if (l_el[1] < l_sh[1] + 0.05) or (r_el[1] < r_sh[1] + 0.05):
            status, issue = "wrong", "Elbow lifted too high (cheating)"
        # curl should not become overhead
        if (l_wr[1] < l_sh[1] - 0.03) or (r_wr[1] < r_sh[1] - 0.03):
            status, issue = "wrong", "Wrist too high (not curl form)"
        # if arms never bend, it's not curl
        if max(l_elbow, r_elbow) > 175:
            status, issue = "wrong", "Arms too straight (no curl)"

    elif exercise == "Shoulder Press":
        wrists_above = (l_wr[1] < l_sh[1] - 0.03) or (r_wr[1] < r_sh[1] - 0.03)
        if not wrists_above:
            status, issue = "wrong", "Press not overhead enough"
        # if torso leans too much while pressing
        if torso_lean > 60:
            status, issue = "wrong", "Leaning too much while pressing"

    return status, issue


# -------------------- Firestore writer --------------------
def send_to_firebase(status, exercise, issue):
    db.collection("postureLogs").document("latest").set(
        {
            "status": status,
            "exercise": exercise,
            "issue": issue,
            "timestamp": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )

    if status == "wrong":
        db.collection("postureHistory").add(
            {
                "status": "wrong",
                "exercise": exercise,
                "issue": issue,
                "timestamp": firestore.SERVER_TIMESTAMP,
            }
        )


# -------------------- Worker thread --------------------
def worker_loop():
    global running, cap, latest_jpeg

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        running = False
        return

    last_sent_key = None  # (exercise, status, issue) to reduce spam

    with mp_pose.Pose(model_complexity=1) as pose:
        while running:
            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Defaults
            exercise = "Squat"
            status = "correct"
            issue = "—"

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Draw skeleton
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Detect exercise + form
                exercise = classify_exercise(lm)
                status, issue = check_form(exercise, lm)

                # Overlay text on frame
                cv2.putText(frame, f"Exercise: {exercise}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Status: {status}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0) if status == "correct" else (0, 0, 255), 2)
                cv2.putText(frame, f"Issue: {issue}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Send to Firestore only when something changes (reduces writes)
            key = (exercise, status, issue)
            if key != last_sent_key:
                send_to_firebase(status, exercise, issue)
                last_sent_key = key

            # Encode frame to JPEG for /video stream
            ok2, jpg = cv2.imencode(".jpg", frame)
            if ok2:
                with lock:
                    latest_jpeg = jpg.tobytes()

            time.sleep(0.01)

    if cap:
        cap.release()
    cap = None


# -------------------- API endpoints --------------------
@app.get("/status")
def api_status():
    return {"running": running}


@app.post("/start")
def api_start():
    global running, worker_thread
    if running:
        return {"ok": True, "msg": "Already running"}

    running = True
    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()
    return {"ok": True, "msg": "AI detection started"}


@app.post("/stop")
def api_stop():
    global running
    running = False
    return {"ok": True, "msg": "AI detection stopped"}


def mjpeg_generator():
    while True:
        if not running:
            time.sleep(0.1)
            continue

        with lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.01)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.get("/video")
def api_video():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
