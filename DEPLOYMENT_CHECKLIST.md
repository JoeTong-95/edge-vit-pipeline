# Repository Structure & Deployment Checklist

**Last Updated:** 2026-04-16  
**Status:** Ready for deployment

---

## Final Repository Layout

### Active Deployment

```
edge-vit-pipeline/
├── DEPLOYMENT_GUIDE.md                    ✓ Read this first
├── README.md (original)
├── LICENSE
├── benchmark.py                           ✓ Performance testing
├── initialize_pipeline.py                 ✓ Main entry point
├── check_dependencies.py                  ✓ Pre-flight check
│
├── docker/
│   ├── requirements.txt                   ✓ Production deps
│   ├── requirements.dev.txt               (optional)
│   └── DOCKER-README.md
│
├── src/
│   ├── configuration-layer/
│   │   ├── config.jetson.vlm-smolvlm-256m.yaml      ✓ ACTIVE JETSON CONFIG
│   │   ├── config.jetson.vlm-smolvlm-256m-trt.yaml  (alt: TRT vision)
│   │   ├── config.yaml                              (default fallback)
│   │   ├── config_schema.py, config_defaults.py, config_types.py, 
│   │   ├── config_normalizer.py, config_validator.py
│   │   ├── config_node.py, config_loader.py
│   │   ├── test/test_config_node.py
│   │   ├── CHANGES.md
│   │   └── README.md
│   │
│   ├── vlm-layer/
│   │   ├── SmolVLM-256M-Instruct/         ✓ ACTIVE VLM MODEL (auto-download)
│   │   │   ├── model.safetensors, config.json, processor_config.json
│   │   │   └── tokenizer.json, special_tokens_map.json, ...
│   │   ├── layer.py                      ✓ VLM inference + TRT vision patch
│   │   ├── util/
│   │   │   ├── visualize_vlm.py
│   │   │   ├── visualize_vlm_realtime.py (async VLM worker)
│   │   │   ├── run_deferred_vlm_queue.py
│   │   │   └── vlm_deferred_queue.py
│   │   ├── test/
│   │   │   ├── smoke_test.py
│   │   │   ├── test_vlm_deferred_queue.py
│   │   │   ├── run_config_vlm_once.py
│   │   │   ├── run_vlm_truck_benchmark.py
│   │   │   └── measure_vlm_modes.py
│   │   ├── CHANGES.md
│   │   └── README.md
│   │
│   ├── yolo-layer/
│   │   ├── models/
│   │   │   ├── yolov11v28_jingtao.engine      ✓ ACTIVE YOLO (FP16, workspace=256)
│   │   │   ├── yolov11v28_jingtao.onnx
│   │   │   ├── yolov8n.pt, yolov11n.pt
│   │   │   └── ...
│   │   ├── detector.py
│   │   ├── test/test_yolo_layer.py
│   │   ├── class_map.py
│   │   ├── TAG_FILTER_BEHAVIOR.md
│   │   ├── CHANGES.md
│   │   └── README.md
│   │
│   ├── tracking-layer/          ✓ ByteTrack
│   ├── roi-layer/               ✓ Vehicle ROI gating
│   ├── vehicle-state-layer/     ✓ State management
│   ├── vlm-frame-cropper-layer/ ✓ Crop extraction
│   ├── scene-awareness-layer/   (optional)
│   ├── metadata-output-layer/   (JSONL logging)
│   ├── evaluation-output-layer/ (optional)
│   ├── input-layer/             (video/camera input)
│   └── ... (other layers)
│
├── scripts/
│   ├── build_trt_smolvlm_vision.py        (optional: vision encoder TRT)
│   └── start_llamacpp_server.sh           (E2B alternative)
│
├── pipeline/
│   ├── README.md
│   ├── pipeline_layers_and_interactions.md
│   ├── CHANGES.md
│   └── codex_ground_rules.md
│
├── data/
│   ├── upson1.mp4                         ✓ Test video
│   └── ...
│
└── _archive/                              (NOT FOR DEPLOYMENT)
    ├── experimental-models/
    │   ├── Qwen3.5-0.8B/
    │   ├── Qwen3.5-0.8B-W4A16-AutoRound-{GPTQ,AWQ}/
    │   ├── Qwen3.5-0.8B-bnb-4bit/
    │   ├── SmolVLM-256M-Instruct-int8/
    │   ├── gemma-4-e2b-{it,gguf}/
    │   └── README.txt (why archived)
    ├── dev-scripts/
    │   ├── benchmark_smolvlm_realtime.py
    │   ├── build_trt_int8.py
    │   └── README.txt
    ├── test-profiles/
    └── yolov11v28_jingtao.engine.bak
```

---

## Deployment Checklist

### Pre-Flight (Before first run on a new Jetson)

- [ ] **OS & Runtime**: Jetson running JetPack 6.x (L4T 36.x), Python 3.10+
- [ ] **Python deps**: `pip install -r docker/requirements.txt` succeeded
- [ ] **GPU access**: `python3 -c "import torch; print(torch.cuda.is_available())"` → True
- [ ] **Storage**: ~5 GB free (for models + logs)
- [ ] **Video data**: `data/upson1.mp4` present (provided in repo)

### Quick Smoke Test

```bash
cd /home/jetson/Desktop/edge-vit-pipeline

# 1. Check all deps
python3 check_dependencies.py
# Expected: [python] OK, [imports] OK, [data] OK, [vlm] OK

# 2. Run short config test
BENCH_CONFIG_YAML=src/configuration-layer/config.jetson.vlm-smolvlm-256m.yaml \
  timeout 30 python3 benchmark.py 2>&1 | tail -20
# Expected: No crashes, VLM latency ~4-5 s
```

### Production Run

```bash
# Option A: Performance benchmark (60 s measurement)
BENCH_CONFIG_YAML=src/configuration-layer/config.jetson.vlm-smolvlm-256m.yaml \
  python3 benchmark.py

# Option B: Live inference (generates JSONL output)
python3 initialize_pipeline.py --output vehicle_detections.jsonl
```

---

## Configuration Profiles

| Config File | Use Case | VLM | Notes |
|-------------|----------|-----|-------|
| `config.jetson.vlm-smolvlm-256m.yaml` | **Production** | SmolVLM-256M-Instruct BF16 | Tuned for Jetson NvMap constraints |
| `config.jetson.vlm-smolvlm-256m-trt.yaml` | **Alternative** (TRT vision) | SmolVLM + TRT vision encoder FP16 | Requires vision encoder TRT build |
| `config.yaml` | Development | Qwen3.5-0.8B | Falls back to CPU if GPU unavailable |

---

## Key Files to Understand

### For Deployment

1. **`DEPLOYMENT_GUIDE.md`** → Full setup + performance baseline
2. **`src/configuration-layer/config.jetson.vlm-smolvlm-256m.yaml`** → Tuning knobs
3. **`benchmark.py`** → Performance testing entry point
4. **`initialize_pipeline.py`** → Production inference entry point

### For Understanding Architecture

1. **`pipeline/pipeline_layers_and_interactions.md`** → How layers connect
2. **`src/vlm-layer/layer.py`** → VLM inference implementation
3. **`branch-optimizer-log.md`** → Detailed session notes (TRT failures, fixes, discoveries)

### For Troubleshooting

1. **`SMOLVLM_OPTIMIZATION_SUMMARY.md`** → Why INT8/torch.compile didn't work
2. **`REALTIME_BENCHMARK_FINDINGS.md`** → Why parallel YOLO+VLM crashes; async architecture
3. **`.gitignore`** → What files are excluded from version control

---

## What's NOT in Deployment

### Archived (in `_archive/`)

- **Experimental VLM models**: Qwen, Gemma variants (high memory, not suitable for Nano)
- **INT8 builders**: `build_trt_int8.py` (explored but not recommended)
- **Realtime test script**: `benchmark_smolvlm_realtime.py` (for development only)
- **Engine backups**: `yolov11v28_jingtao.engine.bak` (old workspace default)

### Why Archived

- **Experimental models** exceeded NvMap or had backend compatibility issues
- **INT8 paths** offered <10% gain on bandwidth-bound Orin Nano workload
- **Dev scripts** were used to debug; production uses `benchmark.py` and `initialize_pipeline.py`

---

## Performance Expectations

| Scenario | Expected | Notes |
|----------|----------|-------|
| **First run** | 10–15 s startup (model download + init) | One-time cost |
| **Steady-state FPS** | 15–18 fps | YOLO real-time, VLM async |
| **VLM latency** | 4–5 s per query | ~1 per minute on this test clip |
| **GPU memory** | ~700 MB stable | YOLO 350 MB + SmolVLM 350 MB + overhead |
| **No crashes** | Yes (after NvMap workspace fix) | Stable production ready |

---

## Git History (This Branch)

```
jetson-optimization-vlm-smolvlm-256m
│
├─ e2f9552 fix(nvmap): rebuild YOLO TRT engine with 256 MB workspace
│          → Resolved NvMap OOM crashes; VLM now stable at 4.9 s/query
│
├─ 2b14a42 perf(vlm): configurable max_new_tokens, SDPA on CUDA
│          → 16% latency improvement; VLM down to 4.1 s/query
│
└─ [main merged upstream]
```

---

## Next Steps After Deployment

1. **Monitor first week**: Check for memory leaks, thermal issues, unexpected slowdowns
2. **Tune dispatch policy**: Adjust `config_vlm_crop_cache_size` based on real traffic
3. **Consider async**: If real-time YOLO (30 fps) matters, switch to `config_vlm_runtime_mode: async`
4. **Evaluate INT8**: If latency is unacceptable and INT8 backend becomes stable on Jetson

---

## Support

- **This repo**: Edge VIT Pipeline (vehicle detection + semantic VLM)
- **NVIDIA Jetson**: https://developer.nvidia.com/embedded/jetson
- **Ultralytics YOLO**: https://github.com/ultralytics/ultralytics
- **HuggingFace SmolVLM**: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct

