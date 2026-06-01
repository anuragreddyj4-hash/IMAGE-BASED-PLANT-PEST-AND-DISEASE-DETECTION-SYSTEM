"""
Unified YOLOv8 Pest & Plant Detection
Usage:
    python unified_detect.py --module pest  --source pest_module/test_image.jpg
    python unified_detect.py --module plant --source plant_module/apple.mp4
    python unified_detect.py --module pest  --source 0          # webcam
"""

import argparse
import os
from ultralytics import YOLO

MODULE_MODELS = {
    "pest":  "pest_module/models/best.pt",
    "plant": "plant_module/models/best.pt",
}

def run(module: str, source: str, conf: float = 0.25):
    model_path = MODULE_MODELS.get(module)
    if not model_path:
        raise ValueError(f"Unknown module '{module}'. Choose from: {list(MODULE_MODELS)}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nPlace best.pt in {os.path.dirname(model_path)}/")

    print(f"[INFO] Loading {module} model: {model_path}")
    model = YOLO(model_path)
    print(f"[INFO] Running detection on: {source}")
    model(source, show=True, conf=conf)
    print("[INFO] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Pest & Plant Detector")
    parser.add_argument("--module", required=True, choices=["pest", "plant"],
                        help="Detection module to use")
    parser.add_argument("--source", required=True,
                        help="Path to image/video, or 0 for webcam")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()
    run(args.module, args.source, args.conf)
