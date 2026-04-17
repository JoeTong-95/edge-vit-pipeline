# Archive — Experimental & Development Files

**Last Updated**: 2026-04-16

This directory contains files and models that are **not** part of the active deployment on Jetson Orin Nano. They are preserved for reference, debugging, and future evaluation.

## Contents

### `experimental-models/`

Alternative VLM checkpoints tested during optimization:

| Model | Why Archived | Size | Notes |
|-------|-------------|------|-------|
| `Qwen3.5-0.8B/` | Full-precision too large for Nano + NvMap issues | 1.8 GB | High-quality 0.8B model; would need INT8 quantization |
| `Qwen3.5-0.8B-W4A16-{GPTQ,AWQ}/` | Quantization backends incompatible with Jetson | ~400 MB | GPTQ/AWQ libraries have Jetson-specific build issues |
| `Qwen3.5-0.8B-bnb-4bit/` | BitsAndBytes 4-bit CUDA kernels crash on Orin | ~300 MB | `bitsandbytes` has CUDA symbol resolution issues on Jetson |
| `SmolVLM-256M-Instruct-int8/` | INT8 dynamic quantization showed <10% gain | ~130 MB | Not worth complexity; BF16 path is simpler & more stable |
| `gemma-4-e2b-{it,gguf}/` | Requires `llama.cpp` server (separate process) | 2–3 GB | Viable but adds deployment complexity; SmolVLM chosen instead |

**Decision**: SmolVLM-256M-Instruct (BF16, PyTorch) is the minimal, stable choice for Jetson Orin Nano with unified memory constraints.

### `dev-scripts/`

Build and profiling scripts from development/exploration:

| Script | Purpose | Status |
|--------|---------|--------|
| `benchmark_smolvlm_realtime.py` | Test YOLO + SmolVLM in parallel queue | Exploratory; findings in `REALTIME_BENCHMARK_FINDINGS.md` |
| `build_trt_int8.py` | Build YOLO TRT INT8 engine | Reference only; INT8 was slower on SM 8.7 at batch=1 |

**Decision**: Deployment uses `benchmark.py` (standard) and `initialize_pipeline.py` (live). These scripts were development-only.

### `test-profiles/`

(Empty placeholder for future per-component benchmarks)

### `yolov11v28_jingtao.engine.bak`

Backup of YOLO TRT engine built with default workspace (1025 MiB). **Do not use** — causes NvMap exhaustion and VLM crashes. Kept only as a reference to the problem it solved.

---

## Why Archive?

Keeping experimental files in the main repo:
- ❌ Adds clutter and confusion about what is "active"
- ❌ Slows down initial clones (large model files)
- ❌ May lead developers to accidentally use outdated configs

Archiving:
- ✓ Keeps repo clean for deployment
- ✓ Preserves history for troubleshooting
- ✓ Speeds up setup for new deployments
- ✓ Allows future reference if requirements change

---

## If You Need to Revisit

### For INT8 on Jetson

1. Check if `bitsandbytes` or `auto-gptq` now support JetPack 6.x
2. Rebuild `SmolVLM-256M-Instruct-int8/` checkpoint with current tools
3. Compare latency to active BF16 path; likely still not worth it on batch=1

### For Larger Models

1. Test `Qwen3.5-0.8B` after quantization (GPTQ if backend available)
2. Implement async VLM queue to hide latency
3. Monitor NvMap fragmentation closely

### For Parallel Inference

See `REALTIME_BENCHMARK_FINDINGS.md` for why YOLO + VLM direct GPU coexistence fails and why async architecture is recommended.

---

## Recovering Files

To restore a file from archive (e.g., for debugging):

```bash
# Restore a model
cp -r _archive/experimental-models/Qwen3.5-0.8B src/vlm-layer/

# Restore a script
cp _archive/dev-scripts/benchmark_smolvlm_realtime.py .
```

---

## Contact / Questions

If you're considering reviving an archived file, check:

1. **`DEPLOYMENT_GUIDE.md`** — Why active config was chosen
2. **`SMOLVLM_OPTIMIZATION_SUMMARY.md`** — Why INT8 / other paths were rejected
3. **`branch-optimizer-log.md`** — Detailed session notes with error messages & decisions

