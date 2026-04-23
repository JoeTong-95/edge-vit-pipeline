# YOLOv11n 6-Class Vehicle Detection Model

Finetuned YOLOv11-nano for traffic surveillance vehicle detection.

## Files

| File | Size | Usage |
|------|------|-------|
| `yolo11n_6class_finetuned.pt` | 5.3 MB | PyTorch (ultralytics) |
| `yolo11n_6class_finetuned.onnx` | 11 MB | ONNX (for TensorRT conversion) |

## Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | car | Sedans, SUVs |
| 1 | pickup | Pickup trucks |
| 2 | van | Vans, RV/motorhomes |
| 3 | truck | Semi, box, dump, tow, utility, garbage, step van, flatbed, concrete mixer, construction |
| 4 | bus | Transit, school, coach, minibus |
| 5 | motorcycle | Motorcycles (no training data yet) |

## Performance (val set, 1313 images)

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|-----------|--------|-------|----------|
| all | 9625 | 0.655 | 0.605 | 0.652 | 0.488 |
| car | 5659 | 0.769 | 0.741 | 0.798 | 0.545 |
| pickup | 706 | 0.507 | 0.396 | 0.441 | 0.332 |
| van | 448 | 0.588 | 0.488 | 0.528 | 0.425 |
| truck | 2571 | 0.746 | 0.779 | 0.820 | 0.607 |
| bus | 241 | 0.667 | 0.623 | 0.672 | 0.532 |

## Training Data

- **4801 images** (3488 train / 1313 val) from two sources:
  - 3562 images from Cornell/NYSDOT highway cameras (9-class, remapped to 6)
  - 1239 frames extracted from TRAFFIC2026Benchmark v4 dataset (47 source runs)
- Base model: YOLOv11n COCO pretrained
- 30 epochs, imgsz=640, batch=16, GPU: RTX A6000

## TensorRT Conversion (Jetson)

TensorRT engines are GPU-architecture-specific. Convert on the target device:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolo11n_6class_finetuned.onnx \
  --saveEngine=yolo11n_6class.engine \
  --fp16
```

Tested target: Jetson Orin Nano 8GB, JetPack 6.2.

## Usage (PyTorch)

```python
from ultralytics import YOLO
model = YOLO("yolo11n_6class_finetuned.pt")
results = model.predict("image.jpg", imgsz=640, conf=0.25)
```

## Date

2026-04-21
