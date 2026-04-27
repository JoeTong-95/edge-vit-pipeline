# GRACE — FHWA Vehicle Classification (Inference Package)

**GRACE** = Geometrically-grounded Rotated Axle Classification Engine

A 44M-parameter ConvNeXtV2 + Gaussian Transformer model that classifies vehicles
into FHWA classes (2–14) from a single cropped image.

## What It Does

Given a truck/vehicle crop (224x224 RGB), GRACE predicts:

| Output | Classes | Description |
|--------|---------|-------------|
| **FHWA class** | 13 (FHWA 2–14) | Federal Highway Administration vehicle class |
| **Vehicle type** | 21 | Semantic type (sedan, semi_tractor, dump_truck, etc.) |
| **Axle count** | 0–12 | Estimated axle count (ordinal regression) |
| **Trailer count** | 4 (0/1/2/3+) | Number of trailers |

Architecture: ConvNeXtV2-Base backbone → Rotated Gaussian Axle Head → Gaussian Transformer → Hierarchical classification (Axle/Trailer → FHWA → Vehicle Type).

## Directory Structure

```
GRACE_inference_package/
├── inference.py            # Main inference script (CLI + importable)
├── config.yaml             # Model architecture config
├── requirements.txt        # Python dependencies
├── checkpoint/
│   └── best_axle_graph_v6.pt   # Model weights (1.2 GB)
├── grace/                  # Model source code
│   ├── __init__.py
│   ├── models/
│   │   ├── axle_graph_v26.py          # Main model (hierarchical V26)
│   │   ├── axle_graph_v25.py          # Parent class (Gaussian Transformer)
│   │   ├── gaussian_head_rotated.py   # Rotated 2D Gaussian head
│   │   ├── gaussian_transformer.py    # Transformer over axle tokens
│   │   └── multiscale_gnn.py          # Multi-scale graph network
│   └── losses/
│       └── ordinal.py                 # Ordinal regression utilities
└── examples/               # Put sample crops here to test
```

## Quick Start

### 1. Install Dependencies

**On Jetson (JetPack 6.x):**

PyTorch and torchvision are pre-installed by JetPack. Only install the extras:

```bash
pip3 install timm pyyaml Pillow
```

**On desktop/server:**

```bash
pip install -r requirements.txt
# Plus: torch torchvision (from https://pytorch.org)
```

### 2. Run Inference

```bash
# Single image
python inference.py path/to/truck_crop.jpg

# Multiple images
python inference.py img1.jpg img2.jpg img3.jpg

# Directory of images (batch processing)
python inference.py --dir path/to/crops/ --batch-size 16

# Save results to JSON
python inference.py --dir path/to/crops/ --output results.json

# Force CPU (no GPU)
python inference.py --device cpu path/to/truck_crop.jpg
```

### 3. Use as Python Library

```python
from inference import load_model, predict_single, predict_batch

# Load model (once)
model, device = load_model()

# Single prediction
result = predict_single(model, device, "truck_crop.jpg")
print(result)
# {'fhwa_class': 'FHWA-9', 'vehicle_type': 'semi_tractor', 'axle_count': 4.97, 'trailer_count': '1', ...}

# Batch prediction
results = predict_batch(model, device, ["img1.jpg", "img2.jpg"], batch_size=8)
```

## Jetson Deployment Notes

### Supported Platforms

- **Jetson AGX Orin (64GB)** — full batch inference, batch_size=16+
- **Jetson AGX Orin (32GB)** — batch_size=8–16
- **Jetson Orin NX (16GB)** — batch_size=4–8
- **Jetson Orin Nano (8GB)** — batch_size=1–2, may need FP16

### JetPack Compatibility

Tested with JetPack 6.x (L4T R36). Key version requirements:

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| JetPack | 6.0 | 6.1+ |
| CUDA | 12.2 | 12.6 |
| PyTorch | 2.1 | 2.4+ |
| Python | 3.10 | 3.11 |

### FP16 Inference (Recommended on Jetson)

For faster inference with minimal accuracy loss:

```python
model, device = load_model()
model = model.half()  # Convert to FP16

# Preprocess as usual, but convert to half precision
img = preprocess("truck.jpg").half().to(device)
with torch.no_grad():
    outputs = model(img, return_heatmap=False)
```

### TensorRT Export (Optional, Advanced)

For maximum throughput on Jetson, export to TensorRT:

```python
import torch

model, device = load_model()
model = model.half()
dummy = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float16)

# Export to ONNX first
torch.onnx.export(
    model, dummy, "grace.onnx",
    input_names=["image"],
    output_names=["fhwa_logits", "primary_logits", "trailer_logits", "axle_count_logits"],
    dynamic_axes={"image": {0: "batch"}},
    opset_version=17,
)
# Then use trtexec to convert:
# trtexec --onnx=grace.onnx --saveEngine=grace.trt --fp16
```

### Memory Optimization

If memory is tight:

```python
# Disable heatmap reconstruction (saves ~10% memory)
with torch.no_grad():
    outputs = model(img, return_heatmap=False)

# Use torch.cuda.amp for automatic mixed precision
with torch.cuda.amp.autocast():
    outputs = model(img, return_heatmap=False)
```

## Model Details

| Property | Value |
|----------|-------|
| Architecture | ConvNeXtV2-Base + Gaussian Transformer V26 |
| Parameters | 44M |
| Input | 224 x 224 RGB, ImageNet normalization |
| Backbone | `convnextv2_base.fcmae_ft_in22k_in1k` (via timm) |
| Checkpoint size | 1.2 GB |
| Inference (A6000) | ~3ms/image (batch=16) |
| Inference (Orin 64GB, FP16) | ~8ms/image (estimated, batch=16) |

### FHWA Class Reference

| Index | Class | Description | Axles | Trailers |
|-------|-------|-------------|-------|----------|
| 0 | FHWA-2 | Passenger cars | 2 | 0 |
| 1 | FHWA-3 | Pickups, vans, SUVs | 2 | 0 |
| 2 | FHWA-4 | Buses | 2–4 | 0 |
| 3 | FHWA-5 | 2-axle 6-tire single unit | 2 | 0 |
| 4 | FHWA-6 | 3-axle single unit | 3 | 0 |
| 5 | FHWA-7 | 4+ axle single unit | 4+ | 0 |
| 6 | FHWA-8 | 3-4 axle single trailer | 2–4 | 1 |
| 7 | FHWA-9 | 5-axle single trailer | 5 | 1 |
| 8 | FHWA-10 | 6+ axle single trailer | 6+ | 1 |
| 9 | FHWA-11 | 5- axle multi-trailer | 3–5 | 2–3 |
| 10 | FHWA-12 | 6-axle multi-trailer | 6 | 2–3 |
| 11 | FHWA-13 | 7+ axle multi-trailer | 7+ | 2–3 |
| 12 | FHWA-14 | (reserved) | — | — |

## Troubleshooting

**`timm` backbone download fails on air-gapped Jetson:**
The model needs `convnextv2_base` weights from timm on first load. On air-gapped systems,
pre-download the timm cache on a connected machine, then copy `~/.cache/huggingface/hub/` to the Jetson.
Alternatively, set `pretrained: false` in `config.yaml` (already set — weights come from our checkpoint).

**CUDA out of memory:**
Reduce `--batch-size` or use `--device cpu`.

**Slow first inference:**
First forward pass compiles CUDA kernels. Subsequent calls are fast. Run a warmup:
```python
model(torch.randn(1, 3, 224, 224, device=device))  # warmup
```

## Contact

Cornell EERL Lab — jg2337@cornell.edu
