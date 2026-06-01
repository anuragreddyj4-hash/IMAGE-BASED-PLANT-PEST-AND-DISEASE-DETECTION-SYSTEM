"""
Real-time pest detection using webcam.
Usage: python pest_module/real_time_detect.py
Press 'q' to quit.
"""

import os
import cv2
from ultralytics import YOLO

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check your camera connection.")

print("[INFO] Webcam started. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        print("[WARN] Failed to read frame.")
        break

    results        = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("Pest Detection - Live", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam released.")
