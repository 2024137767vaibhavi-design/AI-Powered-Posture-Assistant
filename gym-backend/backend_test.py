import time
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

print("✅ Connected to Firestore!")

while True:
    db.collection("postureLogs").document("latest").set({
        "status": "wrong",
        "exercise": "Squat",
        "issue": "Back bending (TEST)",
        "timestamp": firestore.SERVER_TIMESTAMP
    }, merge=True)

    print("Sent: wrong")
    time.sleep(3)

    db.collection("postureLogs").document("latest").set({
        "status": "correct",
        "exercise": "Squat",
        "issue": "—",
        "timestamp": firestore.SERVER_TIMESTAMP
    }, merge=True)

    print("Sent: correct")
    time.sleep(3)
