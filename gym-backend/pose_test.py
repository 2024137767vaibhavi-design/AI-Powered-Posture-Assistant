import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hides INFO + WARNING from TF Lite
os.environ["GLOG_minloglevel"] = "2"       # hides mediapipe glog warnings
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_lite.task"

# Simple skeleton connections (MediaPipe Pose landmark indices)
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),                      # torso
    (23, 25), (25, 27), (24, 26), (26, 28),            # legs
    (27, 31), (28, 32),                                 # feet
    (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),     # face (light)
]

# Create pose landmarker in VIDEO mode
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)  # try 1 if needed
start_time = time.time()

def to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)

    result = detector.detect_for_video(mp_image, timestamp_ms)

    if result.pose_landmarks:
        lms = result.pose_landmarks[0]  # list of 33 landmarks

        # Draw points
        for i, lm in enumerate(lms):
            if getattr(lm, "visibility", 1.0) < 0.4:
                continue
            x, y = to_px(lm, w, h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # Draw connections
        for a, b in CONNECTIONS:
            la, lb = lms[a], lms[b]
            if getattr(la, "visibility", 1.0) < 0.4 or getattr(lb, "visibility", 1.0) < 0.4:
                continue
            xa, ya = to_px(la, w, h)
            xb, yb = to_px(lb, w, h)
            cv2.line(frame, (xa, ya), (xb, yb), (255, 255, 255), 2)

        cv2.putText(frame, "POSE DETECTED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Pose Landmarker - Skeleton", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
