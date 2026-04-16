#!/bin/bash
# Quick GPU test: try a single YOLO inference (no warmup) on smallest model
# Exit 0 if GPU works, 1 if fails
set -e
python3 - <<PYEOF
import torch
from ultralytics import YOLO
import sys

# GPU check
if not torch.cuda.is_available():
    print("✗ CUDA unavailable")
    sys.exit(1)
print(f"✓ CUDA available, VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024 // 1024} MiB")

# Try smallest YOLO (v8n = 6.3 MB)
model = YOLO("/app/src/yolo-layer/models/yolov8n.pt")
print(f"✓ Model loaded: {model.model_name}")

# Single frame (480x360) inference without warmup
import numpy as np
frame = np.zeros((360, 480, 3), dtype=np.uint8)
print("Attempting inference...")
results = model(frame, verbose=False, conf=0.4, device=0)
print(f"✓ Inference succeeded: {len(results[0].boxes)} detections")
sys.exit(0)
PYEOF
