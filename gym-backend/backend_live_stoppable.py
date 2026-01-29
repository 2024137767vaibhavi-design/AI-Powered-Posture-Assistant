import cv2
import mediapipe as mp
import os
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

# --------- Firebase ----------
cred = credentials.Certificate("serviceAccountKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# --------- MediaPipe ----------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

STOP_FILE = "STOP_BACKEND.txt"

def angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def send_to_firebase(status, exercise, issue):
    db.collection("postureLogs").document("latest").set({
        "status": status,
        "exercise": exercise,
        "issue": issue,
        "timestamp": firestore.SERVER_TIMESTAMP
    }, merge=True)

    if status == "wrong":
        db.collection("postureHistory").add({
            "status": "wrong",
            "exercise": exercise,
            "issue": issue,
            "timestamp": firestore.SERVER_TIMESTAMP
        })

def main():
    # remove stop file at start
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not opened")
        return

    last_status = None

    with mp_pose.Pose(model_complexity=1) as pose:
        while True:
            # stop requested?
            if os.path.exists(STOP_FILE):
                print("🛑 Stop requested. Closing backend...")
                break

            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            status = "correct"
            issue = "—"
            exercise = "Squat"

            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                lm = results.pose_landmarks.landmark
                hip = (lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y)
                knee = (lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y)
                ankle = (lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y)

                knee_angle = angle(hip, knee, ankle)

                if knee_angle < 70:
                    status = "wrong"
                    issue = f"Knee too bent ({knee_angle:.0f})"
                elif knee_angle > 160:
                    status = "wrong"
                    issue = f"Leg too straight ({knee_angle:.0f})"

                cv2.putText(frame, f"Status: {status}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0,255,0) if status=="correct" else (0,0,255), 2)
                cv2.putText(frame, f"Issue: {issue}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("AI Gym Backend (Press ESC)", frame)

            if status != last_status:
                send_to_firebase(status, exercise, issue)
                last_status = status

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
