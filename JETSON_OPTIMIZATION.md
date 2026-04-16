# Jetson Optimization — Session Record
**Branch**: `jetson-optimization-A` → merged into `jetson-dev`  
**Device**: NVIDIA Jetson Orin Nano Super (L4T R36.4.7, JetPack 6.x)  
**Sessions**: 2 (April 2026)

---

## Hardware Spec

| Item | Value |
|------|-------|
| Device | Jetson Orin Nano Super Developer Kit |
| GPU | NVIDIA Ampere SM 8.7, 1024 CUDA cores, 32 tensor cores |
| GPU clock | 1020 MHz (MAXN_SUPER / 25W mode) |
| Memory | 8 GB LPDDR5 unified (CPU + GPU share same pool) |
| Memory bandwidth | 102 GB/s |
| AI performance | 67 INT8 TOPS (Sparse) |
| CUDA | 12.6 (L4T driver 540.4.0) |
| TensorRT | 10.3.0 |
| PyTorch | 2.8.0 (Jetson AI Lab wheel, CUDA 12.6) |
| L4T | R36.4.7 |
| Power mode | MAXN_SUPER (25W) — already at max |

**Quantization ceiling on SM 8.7**: FP16 → INT8 → INT4 (tensor cores).  
FP8 requires SM 8.9+ (Ada Lovelace). FP4 requires SM 10.0+ (Blackwell).

---

## Session 1 — Docker / Environment (Blocked)

**Goal**: Get GPU inference running inside an NGC Docker container.

### What was tried
1. NGC 24.10 container → blocked: host driver 540.4.0 requires driver 560+ for that image
2. NGC 24.06 container → CUDA available, but `CUDACachingAllocator.cpp:1131` assert on every YOLO forward pass
3. NGC 24.04 container → same allocator assert

### Root cause found
Jetson uses **NvMap unified memory**. The PyTorch CUDA caching allocator assumes a discrete-GPU memory model and corrupts internal state when NvMap remaps pages under large allocations. This is a known incompatibility between stock PyTorch builds (even official NGC ones) and Jetson's memory subsystem.

### Outcome
Docker abandoned. Native (non-Docker) PyTorch is the correct path on Jetson because it ships the L4T-patched allocator.

---

## Session 2 — Native Stack + GPU Inference + Optimization

### Step 1: Native PyTorch

**Problem**: PyPI `torch 2.11.0+cu130` compiled for CUDA 13.0. Jetson driver only supports CUDA 12.6. `CUDA available: False`.

**Fix**: Install `torch 2.8.0` from the Jetson AI Lab PyPI mirror (CUDA 12.6 wheel).

```bash
bash docker/setup-native-jetson.sh
```

### Step 2: VLM model files

Git LFS pointers (133 bytes stubs) instead of real weights. Downloaded via `huggingface_hub`:
- `tokenizer.json` (12 MB)
- `model.safetensors-00001-of-00001.safetensors` (1.7 GB) from `Qwen/Qwen3.5-0.8B`

### Step 3: NvMap unified memory workaround

**Problem**: Even native PyTorch hit the NvMap assert during YOLO's first inference after the VLM model loaded.

**Root cause**: Jetson's NvMap carve-out (~1.5 GB usable for GPU) is shared with CPU allocations. Loading the 1.7 GB VLM to CPU fills the carve-out, leaving no room for YOLO's ~80 MB of CUDA activations.

**Fixes**:

| Fix | Where | Effect |
|-----|-------|--------|
| `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` | `benchmark.py` (env var set at import) | Prevents single large NvMap blocks from blocking smaller allocations |
| YOLO GPU warmup in `initialize_yolo_layer` | `src/yolo-layer/detector.py` | Forces YOLO to claim its ~80 MB CUDA allocation *before* VLM loads, reserving the NvMap space |
| `config_vlm_device: cpu` | `config.jetson.yaml` | Keeps VLM off GPU entirely; VLM is async so CPU latency (~107 s/query) is acceptable |
| VLM uses `bfloat16` | `src/vlm-layer/layer.py` | Model's native dtype; avoids float16 conversion on load that could trigger NvMap bugs |

### Step 4: TensorRT FP16 engine

Ultralytics' `.pt` YOLO runs through PyTorch eager inference. Exporting to a TRT engine eliminates PyTorch dispatch overhead and uses optimised FP16 CUDA kernels selected at build time.

**Build process** (run once on Jetson, engine is device-specific):
```bash
# Export ONNX + build TRT FP16 engine (~12 min)
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python3 -c "
import torch, numpy as np
dummy = torch.zeros(1,3,640,640,dtype=torch.float16,device='cuda'); del dummy
from ultralytics import YOLO
m = YOLO('src/yolo-layer/models/yolov11v28_jingtao.pt')
m.predict(np.zeros((640,640,3),dtype=np.uint8), device='cuda', half=True, verbose=False)
m.export(format='engine', device='cuda', half=True, imgsz=640)
"
# Engine saved to: src/yolo-layer/models/yolov11v28_jingtao.engine
```

**Key NvMap trick**: pre-warm CUDA context with a dummy tensor *before* loading YOLO, then warmup YOLO (predict once) *before* export. TRT's calibration/build phase then runs within the already-claimed NvMap budget.

**Note on TRT engine portability**: engine is compiled for SM 8.7, TRT 10.3, CUDA 12.6. Do NOT copy to x86 or other Jetson variants.

### Step 5: INT8 investigation

Built an INT8 engine at 384×640 (rectangular, matches 640×360 source with <6% letterbox waste vs 44% for square 640×640).

**Result**: INT8 was **20% slower** than FP16 in the full pipeline (11.96 fps INT8 vs 14.95 fps FP16).

**Why**: SM 8.7 at batch=1 is **memory-bandwidth bound**, not compute bound. INT8 halves arithmetic operations but the bandwidth cost of moving weight/activation tensors is nearly the same. FP16 tensor cores are faster here.

INT8 becomes useful at batch ≥ 4 where compute starts to dominate. The INT8 build script and engine are kept as reference (`scripts/build_trt_int8.py`, `src/yolo-layer/models/yolov11v28_jingtao_int8.engine`).

### Step 6: Eliminate double YOLO per frame

**Problem found**: `benchmark.py` called `run_yolo_detection` twice per frame:
1. A `dets_boot` call to feed `update_roi_state()` (ROI box collection)
2. The main detection call for the downstream pipeline

After ROI locks, `update_roi_state()` is a **no-op** (returns immediately on line 76 of `roi_layer.py`). The first call was entirely wasted. Before lock, both calls used the same full-frame input, so results were redundant anyway.

**Fix**: single YOLO call per frame; pass those detections to both the pipeline and `update_roi_state()`.

**Result**: 14.95 → **22.24 fps** (+49%).

---

## Performance Progression

| State | YOLO backend | Pipeline FPS | YOLO ms/frame |
|-------|-------------|-------------|---------------|
| Baseline (PyTorch .pt FP32, Docker) | NGC 24.06 .pt | blocked | — |
| Native PyTorch .pt FP16 | `yolov11v28_jingtao.pt` | 7.94 | 59.6 ms |
| + TRT FP16 engine | `yolov11v28_jingtao.engine` | 14.95 | 28.6 ms |
| + Single YOLO per frame | same engine | **22.24** | 36.6 ms |
| INT8 384×640 (tested, reverted) | `yolov11v28_jingtao_int8.engine` | 11.96 | 33.0 ms |

YOLO capacity (engine alone, isolated) is ~35 fps — above the 30 fps source rate. Pipeline is below 30 fps due to remaining overhead (video decode ~7 ms, tracking ~1 ms, other layers).

---

## Benchmark Configs

Three configs are available for performance evaluation:

### 1. PyTorch .pt baseline (FP16, no TRT)
```yaml
# In config.jetson.yaml: change yolo model line to:
config_yolo_model: yolov11v28_jingtao.pt
# (remove config_yolo_imgsz override)
```
Run: `BENCH_CONFIG_YAML=src/configuration-layer/config.jetson.yaml python3 benchmark.py`

### 2. TRT FP16 (current default — best performance)
```yaml
config_yolo_model: yolov11v28_jingtao.engine
config_yolo_imgsz:   # null = model default (640x640)
```
Run: `BENCH_CONFIG_YAML=src/configuration-layer/config.jetson.yaml python3 benchmark.py`

### 3. TRT INT8 384×640 (reference — do not use, slower than FP16 on this device)
```yaml
config_yolo_model: yolov11v28_jingtao_int8.engine
config_yolo_imgsz:
  - 384
  - 640
```
Run: `BENCH_CONFIG_YAML=src/configuration-layer/config.jetson.yaml python3 benchmark.py`

---

## Jetson-Specific Changes vs `main`

### New files
| File | Purpose |
|------|---------|
| `src/configuration-layer/config.jetson.yaml` | Jetson-tuned config (cuda device, TRT engine, VLM on CPU) |
| `src/configuration-layer/config.cpu-test.yaml` | CPU-only config for verifying pipeline without GPU |
| `docker/setup-native-jetson.sh` | Reproducible native torch install (Jetson AI Lab wheels) |
| `scripts/build_trt_int8.py` | Standalone TRT INT8 builder using TRT Python API + torch GPU buffers |
| `data/calib.yaml` | Calibration dataset YAML for INT8 build (300 frames from 4 videos) |

### Modified files (Jetson-specific logic)
| File | Change |
|------|--------|
| `benchmark.py` | `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`; reads `config_vlm_device` and `config_yolo_imgsz`; single YOLO call per frame |
| `src/yolo-layer/detector.py` | CUDA FP16 inference; GPU warmup before VLM loads; `is_trt` flag skips dynamic ROI imgsz for TRT engines; `forced_imgsz` for non-square engines |
| `src/vlm-layer/layer.py` | Uses `bfloat16` on CUDA (model native dtype); CPU fallback path |
| `src/tracking-layer/tracker.py` | `build_tracking_layer_package` emits both nested and flat-array fields |
| `src/configuration-layer/config*.py` | Added `config_vlm_device` and `config_yolo_imgsz` to schema, defaults, types, normalizer, validator |
| `src/configuration-layer/config.yaml` | Added `config_vlm_device:` and `config_yolo_imgsz:` with null defaults |

### Docker files (legacy, not used for native deployment)
| File | Note |
|------|------|
| `docker/Dockerfile.jetson` | NGC-based image — do not use; NvMap incompatibility |
| `docker/run-docker-jetson` | Kept for reference; was fixed with `--ipc=host` and ulimits |

---

## Architecture Decisions

### VLM on CPU, YOLO on GPU
The 1.7 GB VLM model fills Jetson's NvMap carve-out even when loaded to CPU. Running VLM on GPU would prevent YOLO from getting GPU memory. Since:
- YOLO runs every frame (latency-critical)
- VLM runs asynchronously in a worker thread (throughput over latency)

VLM stays on CPU. Its ~107 s/query latency is acceptable because the async worker queues queries and doesn't block the frame loop.

### TRT FP16 over INT8
On SM 8.7 at batch=1, YOLO inference is memory-bandwidth bound. INT8 reduces arithmetic by 2× but bandwidth savings are minimal for small activation tensors. FP16 tensor cores are the practical ceiling. INT8 would help at batch ≥ 4.

### Square 640×640 over rectangular 384×640
The 640×640 TRT engine is faster than 384×640 INT8 in practice because:
1. Ultralytics' letterbox path for square inputs is more optimised
2. FP16 outweighs the pixel reduction benefit of rectangular inputs at this scale

---

## Known Limitations

- VLM at 107 s/query on CPU is very slow. Future path: bitsandbytes INT8 or GPTQ quantization to fit VLM on GPU alongside YOLO (bitsandbytes is not yet on Jetson AI Lab mirror as of this session).
- Pipeline at 22.24 fps is still below the 30 fps source rate. Remaining overhead is video decode (~7 ms/frame) and frame loop Python overhead.
- TRT engines must be rebuilt if TRT version changes (TRT 10.3 engines are not forward-compatible).
- The `model.safetensors` and `tokenizer.json` VLM files are not in the repo (too large for git). Run `docker/setup-native-jetson.sh` and download weights separately.
