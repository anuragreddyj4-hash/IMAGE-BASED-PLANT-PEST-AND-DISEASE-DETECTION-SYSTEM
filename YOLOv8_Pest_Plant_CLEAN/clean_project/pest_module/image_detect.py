"""
Pest detection on a single image.
Usage: python pest_module/image_detect.py
"""

import os
import cv2
from ultralytics import YOLO

# ── Paths (relative to project root) ────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
IMG_PATH   = os.path.join(BASE_DIR, "test_image.jpg")
OUT_PATH   = os.path.join(BASE_DIR, "output_image.jpg")
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"Test image not found: {IMG_PATH}")

model  = YOLO(MODEL_PATH)
image  = cv2.imread(IMG_PATH)
results = model(image)

for r in results:
    annotated = r.plot()
    cv2.imshow("Pest Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(OUT_PATH, annotated)
    print(f"[INFO] Result saved → {OUT_PATH}")
