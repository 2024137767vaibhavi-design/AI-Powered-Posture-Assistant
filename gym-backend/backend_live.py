import cv2
import mediapipe as mp
from firebase_admin import credentials, firestore
import firebase_admin
from utils import angle

# ----- Firebase Init -----
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ----- MediaPipe Init -----
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

# A simple rule for squat demo:
# Knee angle too small/too large -> wrong
# Back angle check can be added later

def send_to_firebase(status, exercise, issue):
    db.collection("postureLogs").document("latest").set({
        "status": status,
        "exercise": exercise,
        "issue": issue,
        "timestamp": firestore.SERVER_TIMESTAMP
    }, merge=True)

    # Log wrong events to history
    if status == "wrong":
        db.collection("postureHistory").add({
            "status": "wrong",
            "exercise": exercise,
            "issue": issue,
            "timestamp": firestore.SERVER_TIMESTAMP
        })

with mp_pose.Pose(model_complexity=1) as pose:
    last_status = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        status = "correct"
        issue = "—"
        exercise = "Squat"

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Use right leg points (you can also average left+right)
            # Hip, Knee, Ankle
            hip = (lm[mp_pose.PoseLandmark.RIGHT_HIP].x, lm[mp_pose.PoseLandmark.RIGHT_HIP].y)
            knee = (lm[mp_pose.PoseLandmark.RIGHT_KNEE].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE].y)
            ankle = (lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y)

            knee_angle = angle(hip, knee, ankle)

            # Simple threshold example:
            # Good squat knee angle around ~80 to 120 depending on depth.
            if knee_angle < 70:
                status = "wrong"
                issue = f"Knee too bent (angle={knee_angle:.0f})"
            elif knee_angle > 160:
                status = "wrong"
                issue = f"Leg too straight (angle={knee_angle:.0f})"

            # Show on screen
            cv2.putText(frame, f"Status: {status}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if status=="correct" else (0, 0, 255), 2)
            cv2.putText(frame, f"Issue: {issue}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("AI Gym Backend (ESC to quit)", frame)

        # Send update only when status changes (avoid spamming Firebase)
        if status != last_status:
            send_to_firebase(status, exercise, issue)
            last_status = status

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
