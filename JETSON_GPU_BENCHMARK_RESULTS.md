# Jetson GPU Benchmark Results
**Branch**: `jetson-optimization-A`  
**Date**: 2026-04-16  
**Device**: Jetson Orin Nano (L4T R36.4.7, JetPack 6.x, CUDA 12.6)  
**Stack**: PyTorch 2.8.0 (Jetson CUDA 12.6 wheel), native (no Docker)

---

## Pipeline Configuration
| Component | Setting |
|-----------|---------|
| YOLO model | `yolov11v28_jingtao.pt` on `cuda` (FP16) |
| VLM model | `Qwen3.5-0.8B` on `cpu` (async worker) |
| Frame resolution | 640×360 |
| ROI enabled | ✅ threshold=3 (locked at frame 293) |
| cuDNN benchmark | ✅ (`torch.backends.cudnn.benchmark=True`) |

---

## Benchmark Results (60s measured window)

| Metric | Value |
|--------|-------|
| **End-to-end pipeline FPS** | **7.94 fps** |
| YOLO GPU capacity | 16.8 fps (59 ms/frame) |
| VLM async latency (CPU) | ~107 s/query |
| ROI lock | frame 293, 0.1% of frame area |
| Frames processed | 477 |
| Avg detections/frame | 0.13 |

### Per-layer timing (avg ms/frame)
| Layer | Time |
|-------|------|
| Input (video decode) | 7.36 ms |
| ROI | 0.18 ms |
| **YOLO (GPU FP16)** | **59.6 ms** |
| Tracking | 0.78 ms |
| Vehicle state | 0.11 ms |
| VLM submit (async) | 0.11 ms |

**Note**: End-to-end (125 ms) is higher than layer sum (68 ms) because an additional
un-timed YOLO call runs for ROI update each frame — effectively 2×YOLO per frame.

---

## Unblocking Steps Completed This Session

### 1. Python / Torch Stack (native, no Docker)
- **Installed**: `torch 2.8.0` + `torchvision 0.23.0` from Jetson AI Lab mirror
  (`pypi.jetson-ai-lab.io/jp6/cu126`)  
- **Why**: PyPI torch 2.11.0 was compiled for CUDA 13.0; Jetson driver supports CUDA 12.6 only
- **Result**: `CUDA available: True`, Device: `Orin`

### 2. VLM Model Files (Git LFS pointers → real weights)
- `tokenizer.json` (12 MB) and `model.safetensors` (1.7 GB) were LFS pointer stubs
- Downloaded via `huggingface_hub` from `Qwen/Qwen3.5-0.8B`

### 3. Jetson NvMap Unified Memory Workaround
**Root cause**: Jetson's NvMap carve-out (~1.5 GB usable) is shared between CPU and
GPU tensors. Loading a 1.7 GB model (even to CPU) fills the carve-out, preventing
YOLO from claiming GPU memory for its first inference.

**Fixes applied**:
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` — prevents single large NvMap blocks
- YOLO GPU warmup in `initialize_yolo_layer` — claims CUDA memory before VLM loads
- `config_vlm_device: cpu` in `config.jetson.yaml` — VLM on CPU, YOLO keeps full GPU
- VLM uses `bfloat16` (native dtype) instead of `float16` — avoids extra conversion overhead

### 4. Code Fixes
| File | Fix |
|------|-----|
| `src/yolo-layer/detector.py` | Added FP16 inference + explicit GPU warmup |
| `src/vlm-layer/layer.py` | Use `bfloat16` on CUDA (model's native dtype) |
| `src/tracking-layer/tracker.py` | `build_tracking_layer_package` now emits both nested and flat-array formats |
| `src/configuration-layer/config_schema.py` | Added `config_vlm_device` key |
| `src/configuration-layer/config_types.py` | Added `config_vlm_device` field |
| `src/configuration-layer/config_normalizer.py` | Normalizes `config_vlm_device` |
| `src/configuration-layer/config_defaults.py` | Default `config_vlm_device: ""` |
| `src/configuration-layer/config_validator.py` | Validates `config_vlm_device` |
| `benchmark.py` | Reads `config_vlm_device`, sets `max_split_size_mb:128`, wires vlm_device |
| `docker/setup-native-jetson.sh` | New: reproducible native install script |

---

## Next Optimization Opportunities

| Opportunity | Expected Gain | Effort |
|-------------|---------------|--------|
| Fix double YOLO per frame (ROI update uses separate YOLO call) | ~1.8× pipeline FPS | Medium |
| TensorRT FP16 YOLO export (on x86, copy engine to Jetson) | 3-5× YOLO throughput | High |
| VLM quantization (INT8, ~850 MB) → fits on GPU alongside YOLO | VLM GPU: 5-10× faster | High (bitsandbytes not on Jetson AI Lab yet) |
| Increase ROI threshold awareness (video has only ~7 vehicles) | Small | Low |

---

## Reproducibility

```bash
# Install correct torch stack (run once)
bash docker/setup-native-jetson.sh

# Download VLM weights (run once, ~1.7 GB)
python3 -c "
from huggingface_hub import hf_hub_download
import shutil
for f in ['tokenizer.json', 'model.safetensors-00001-of-00001.safetensors']:
    src = hf_hub_download('Qwen/Qwen3.5-0.8B', f)
    shutil.copy2(src, f'src/vlm-layer/Qwen3.5-0.8B/{f}')
"

# Run benchmark
BENCH_CONFIG_YAML=src/configuration-layer/config.jetson.yaml python3 benchmark.py
```
