"""
Pest detection on a video file.
Usage: python pest_module/video_detect.py
Press 'q' to quit early.
"""

import os
import cv2
from ultralytics import YOLO

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best.pt")
VIDEO_PATH  = os.path.join(BASE_DIR, "test_video.mp4")
OUTPUT_PATH = os.path.join(BASE_DIR, "output_video.mp4")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(VIDEO_PATH)

# Read actual video dimensions (avoids hardcoded 1280x720)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print(f"[INFO] Processing: {VIDEO_PATH}  ({width}x{height} @ {fps:.1f} fps)")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results         = model(frame)
    annotated_frame = results[0].plot()

    out.write(annotated_frame)
    cv2.imshow("Pest Detection - Video", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("[INFO] Early exit by user.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[INFO] Output saved → {OUTPUT_PATH}")
