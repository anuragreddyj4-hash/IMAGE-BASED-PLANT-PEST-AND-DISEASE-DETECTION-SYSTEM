# Unified YOLOv8 Pest & Plant Detection

## Project Structure

```
project/
├── unified_detect.py              ← main entry point
├── requirements.txt
├── pest_module/
│   ├── models/
│   │   └── best.pt               ← copy from original pest_module/pest-detection-main/best.pt
│   ├── datasets/
│   │   └── data.yaml
│   ├── test_image.jpg            ← copy from original
│   ├── test_video.mp4            ← copy from original
│   ├── image_detect.py
│   ├── real_time_detect.py
│   └── video_detect.py
└── plant_module/
    ├── models/
    │   └── best.pt               ← copy from original plant_module/.../best.pt (if trained)
    ├── apple.mp4                 ← copy from original plant_module/.../apple.mp4
    └── track.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

### Pest detection on image
```bash
python pest_module/image_detect.py
```

### Pest detection on video
```bash
python pest_module/video_detect.py
```

### Pest detection via webcam
```bash
python pest_module/real_time_detect.py
```

### Plant tracking on video
```bash
python plant_module/track.py
```

### Unified entry point
```bash
python unified_detect.py --module pest  --source pest_module/test_image.jpg
python unified_detect.py --module plant --source plant_module/apple.mp4
python unified_detect.py --module pest  --source 0   # webcam
```
