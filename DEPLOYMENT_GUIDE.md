# Deployment Guide — Edge VIT Pipeline (SmolVLM-256M on Jetson Orin Nano)

**Last Updated:** 2026-04-16  
**Branch:** `jetson-optimization-vlm-smolvlm-256m`  
**Status:** Ready for production testing on Jetson Orin Nano Super with unified memory (NvMap).

---

## Overview

This is a **stable, production-oriented configuration** for running the full vehicle detection + VLM semantic classification pipeline on Jetson Orin Nano:

- **YOLO Detection**: Ultralytics YOLO v11 (TensorRT FP16, workspace-optimized)
- **VLM Semantic**: HuggingFace SmolVLM-256M-Instruct (PyTorch BF16 on CUDA with SDPA attention)
- **Real-time target**: ~17 FPS pipeline throughput; ~4.1 s VLM latency per query (typical)

---

## Quick Start

### Prerequisites

1. **Jetson Environment**:
   - Jetson Orin Nano Super (8 GB unified memory)
   - JetPack 6.x (L4T 36.x) with CUDA, cuDNN
   - Python 3.10+

2. **Dependencies** (see `docker/requirements.txt`):
   ```bash
   pip install -r docker/requirements.txt
   ```

3. **Models** (auto-downloaded on first run):
   - SmolVLM-256M-Instruct (HF, ~500 MB)
   - YOLO v11 v28-jingtao TRT engine (built in repo, 52 MB)

### Run the Pipeline

**Option 1: Full end-to-end benchmark** (60 s measurement window):
```bash
BENCH_CONFIG_YAML=src/configuration-layer/config.jetson.vlm-smolvlm-256m.yaml \
  python3 benchmark.py
```

**Option 2: Real-time pipeline with video file** (JSONL output):
```bash
python3 initialize_pipeline.py --output data/pipeline_output.jsonl
```

**Option 3: Live camera test** (real-time display with YOLO + tracking):
```bash
# Must have a camera connected to Jetson
python3 src/input-layer/live_camera_pipeline_test.py

# Or with options:
python3 src/input-layer/live_camera_pipeline_test.py \
  --device 0 \
  --width 1280 --height 720 \
  --conf 0.4 \
  --device-compute cuda

# Controls while running:
#   q = quit
#   s = save screenshot
```

---

## Performance Baseline

**Latest benchmark run (2026-04-16, SmolVLM + YOLO TRT + 256 MB workspace):**

| Metric | Value | Notes |
|--------|-------|-------|
| **Pipeline FPS** | 17.24 fps | Frame rate (YOLO + tracking + VLM inline) |
| **End-to-end latency** | 57.99 ms | Per frame, measured window |
| **VLM latency** | 4,138.8 ms | Per query (avg 1 per 60 s window) |
| **YOLO latency** | 40.53 ms | Per frame on ROI crop |
| **GPU memory usage** | ~36 MiB TRT + ~500+ MiB SmolVLM | Stable, no crashes |

**Comparison to prior runs:**
- **Before workspace fix**: VLM crashes (`CUDACachingAllocator:1131` OOM) due to YOLO TRT using 1025 MiB workspace
- **After workspace fix**: Stable, VLM runs at 4.9 s/query
- **After decode + SDPA optimizations**: VLM at **4.1 s/query** (16% improvement)

---

## Configuration

### Active Production Config

**File**: `src/configuration-layer/config.jetson.vlm-smolvlm-256m.yaml`

Key tuning parameters:

```yaml
config_device: cuda
config_vlm_device: cuda
config_vlm_model: src/vlm-layer/SmolVLM-256M-Instruct
config_vlm_max_new_tokens: 24        # Decode budget (24 tokens for compact JSON)
config_vlm_worker_batch_size: 1      # Batch 1 = min peak activation (NvMap safe)
config_vlm_worker_batch_wait_ms: 0   # Lowest latency wait
config_yolo_model: yolov11v28_jingtao.engine  # TRT FP16, 256 MB workspace
```

### Tuning Knobs (for deployment customization)

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `config_vlm_max_new_tokens` | 24 | 8–256 | Lower → faster worst-case decode; risk JSON truncation |
| `config_vlm_worker_batch_size` | 1 | 1–4 | Higher → more throughput but larger activation peaks |
| `config_yolo_confidence_threshold` | 0.4 | 0.1–0.9 | Higher → fewer detections & VLM queries |

---

## Known Limitations & Workarounds

### 1. NvMap Unified Memory Constraint

**Issue**: YOLO TRT default workspace (1025 MiB) exhausts ~70% of available NvMap IOVMM pool (~1400 MiB), leaving insufficient contiguous memory for concurrent VLM inference.

**Solution Applied**: Rebuilt YOLO TRT engine with `workspace=256` MB at engine build time (reducing TRT footprint to ~350 MB).

**Do not:**
- Increase YOLO batch size significantly (will exhaust memory)
- Load additional TRT models alongside YOLO+VLM without testing

### 2. VLM Latency Not Real-time

**Issue**: 4.1 s/query is too long for interactive real-time UX (want <1 s typically).

**Why**: SmolVLM-256M is a full vision+language model (~260M params total), not a lightweight classifier. Decode inherently takes time.

**Options**:
- Use **async worker mode** (`config_vlm_runtime_mode: async`) so YOLO maintains ~30 FPS while VLM processes in background (see `REALTIME_BENCHMARK_FINDINGS.md`)
- Switch to a **smaller model** (research needed)
- Move to **dedicated VLM server** (TensorRT-LLM–class for future)

### 3. Only 1 VLM Query Per 60 s on This Clip

**Issue**: Sparse semantic dispatch policy means VLM runs infrequently.

**Why**: Crop cache + dead-frame gating mean the pipeline only queries for new vehicle tracks.

**Customization**: Tune `config_vlm_crop_cache_size` and `config_vlm_dead_after_lost_frames` to increase/decrease dispatch frequency.

---

## Architecture & Design

### Pipeline Layers (in order)

1. **Input Layer** → frame decode (video/camera)
2. **ROI Layer** → region of interest (vehicle count gate)
3. **YOLO Layer** → object detection (TensorRT FP16)
4. **Tracking Layer** → ByteTrack (persistent ID assignment)
5. **Vehicle State Layer** → temporal state management
6. **VLM Frame Cropper** → extract crops from tracked objects
7. **VLM Layer** → semantic inference (PyTorch BF16 + SDPA)
8. **Metadata Output Layer** → JSONL logging

### Memory Layout (Jetson Orin Nano Super)

| Component | GPU Memory | CPU Memory | Notes |
|-----------|-----------|-----------|-------|
| CUDA runtime | ~100 MB | — | Unavoidable overhead |
| YOLO TRT engine weights + context | ~50 MB + 36 MB | — | Rebuilt with workspace=256 MB |
| SmolVLM-256M weights (BF16) | ~500 MB (virtual via CUDA vmem API) | — | Not bounded by NvMap IOVMM |
| SmolVLM activations (batch=1) | ~150–200 MB peak (NvMap) | — | Temporary during forward/decode |
| **Total stable load** | **~700 MB** | — | Well within 8 GB unified memory |

---

## Testing & Validation

### Unit Tests (Quick)

```bash
# Config layer
python3 src/configuration-layer/test/test_config_node.py

# Dependencies check
python3 check_dependencies.py

# Camera + YOLO + tracking integration test (live display)
python3 src/input-layer/live_camera_pipeline_test.py
```

### Camera Hardware Setup (Before Live Test)

**For Jetson with CSI camera** (e.g., RPi Camera Module 3):
```bash
# Live camera test auto-uses GStreamer
python3 src/input-layer/live_camera_pipeline_test.py --use-gstreamer

# Expected: Display window showing live camera feed with YOLO detections + tracking IDs
```

**For USB camera**:
```bash
# Detect camera device
ls /dev/video*

# Test with device index (e.g., /dev/video0)
python3 src/input-layer/live_camera_pipeline_test.py --device 0
```

**No camera?** Skip to benchmark with video file:
```bash
BENCH_CONFIG_YAML=src/configuration-layer/config.jetson.vlm-smolvlm-256m.yaml python3 benchmark.py
```

### Full Benchmark (60 s)

```bash
BENCH_CONFIG_YAML=src/configuration-layer/config.jetson.vlm-smolvlm-256m.yaml python3 benchmark.py
```

Expected output:
- No `CUDACachingAllocator` errors
- VLM latency ~3.5–5.0 s/query
- Pipeline FPS ~15–18
- Health check: "likely keeps up" for VLM queue

### Live Pipeline (test with real video)

```bash
python3 initialize_pipeline.py --output /tmp/test_output.jsonl --max-frames 100
```

Check output: `cat /tmp/test_output.jsonl | head -1 | python3 -m json.tool`

---

## Optimization History

### Phase 1: NvMap Stability (PR: `fix(nvmap)`)

- **Problem**: YOLO TRT workspace (default 1025 MiB) + SmolVLM activations → OOM crashes
- **Fix**: Rebuild YOLO engine with `workspace=256` MB
- **Result**: Stable coexistence, VLM latency ~4.9 s/query

### Phase 2: Decode Latency (PR: `perf(vlm)`)

- **Problem**: Hardcoded `max_new_tokens=32` too conservative; SDPA not enabled
- **Fixes**:
  1. Made `max_new_tokens` configurable in YAML (default 32, set to 24 in SmolVLM profile)
  2. Auto-enable `attn_implementation="sdpa"` on CUDA when model supports it
  3. Thread `config_vlm_device` through all init paths
- **Result**: VLM latency improved to ~4.1 s/query (16% gain)

---

## File Organization

### Active Deployment Files

```
src/
├── configuration-layer/
│   ├── config.jetson.vlm-smolvlm-256m.yaml  ← ACTIVE JETSON CONFIG
│   ├── config.jetson.vlm-smolvlm-256m-trt.yaml
│   ├── config.yaml (default)
│   └── ... (config schema, defaults, types, normalizer, validator)
├── vlm-layer/
│   ├── SmolVLM-256M-Instruct/  ← ACTIVE VLM MODEL (auto-downloaded)
│   ├── layer.py  ← VLM inference + TRT integration
│   └── CHANGES.md
├── yolo-layer/
│   ├── models/
│   │   └── yolov11v28_jingtao.engine  ← ACTIVE YOLO (rebuilt 256 MB workspace)
│   └── detector.py
├── tracking-layer/
├── roi-layer/
├── vehicle-state-layer/
├── scene-awareness-layer/
├── metadata-output-layer/
└── evaluation-output-layer/

scripts/
├── build_trt_smolvlm_vision.py  ← Optional: build vision encoder TRT
└── start_llamacpp_server.sh  ← For E2B alternative

_archive/  ← NOT USED IN ACTIVE DEPLOYMENT
├── experimental-models/  (Qwen, Gemma variants)
├── dev-scripts/  (INT8 builders, realtime benchmark)
└── test-profiles/  (deprecated profiling)

benchmark.py  ← Main performance test
initialize_pipeline.py  ← Main inference entry point
check_dependencies.py  ← Quick dependency check
```

### Documentation

```
DEPLOYMENT_GUIDE.md  ← THIS FILE
branch-optimizer-log.md  ← Detailed optimization session notes
SMOLVLM_OPTIMIZATION_SUMMARY.md  ← Optimization attempts & outcomes
REALTIME_BENCHMARK_FINDINGS.md  ← Async architecture recommendations
JETSON_OPTIMIZATION.md  ← Overall Jetson constraints & strategy
docker/requirements.txt  ← Python dependencies
```

---

## Maintenance & Troubleshooting

### If Pipeline Crashes with OOM

**Check 1: GPU Memory Fragmentation**
```bash
python3 -c "import torch; print(torch.cuda.memory_summary())"
```

**Check 2: NvMap State**
```bash
# On Jetson:
cat /proc/nvmap
```

**Solution**: Reboot Jetson to reset NvMap IOVMM state.

### If VLM Inference Slow (>5 s)

1. Check GPU clock scaling: `tegrastats | grep GR3D`  
   (Should see "on" for active GPU; if "off", device may be thermal throttling)
2. Check CUDA memory pressure: increase `config_vlm_worker_batch_size` from 1 to 2, measure again
3. Reduce `config_vlm_max_new_tokens` from 24 to 20 (test JSON quality)

### If YOLO Accuracy Drops

- **Confidence threshold too high**: lower `config_yolo_confidence_threshold` from 0.4 to 0.3
- **ROI gate too tight**: increase `config_roi_vehicle_count_threshold`

---

## Next Steps for Further Optimization

1. **Async VLM**: Run YOLO at full frame rate while VLM processes in background queue (see `src/vlm-layer/util/visualize_vlm_realtime.py`)
2. **INT8 Quantization**: Weight-only INT8 for VLM (~5–15% latency gain if working; not trivial on Jetson)
3. **TRT LLM Path**: Build VLM decoder with TensorRT-LLM (requires x86 host for compilation)
4. **Smaller Model**: Evaluate truly lightweight VLM alternatives if 4 s is unacceptable

---

## Support & References

- **Jetson Docs**: https://docs.nvidia.com/jetson/  
- **Ultralytics YOLO**: https://docs.ultralytics.com/  
- **Hugging Face SmolVLM**: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct  
- **TensorRT**: https://docs.nvidia.com/deeplearning/tensorrt/  

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-04-16 | 1.0 | Initial production release: SmolVLM + YOLO TRT on Jetson Orin Nano (256 MB workspace, SDPA, configurable decode) |

