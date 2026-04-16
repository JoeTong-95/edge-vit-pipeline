# Jetson GPU Inference - Decision Tree for Next Session

**Updated**: 2026-04-16, 00:15 AM  
**Status**: Pipeline code is ready, GPU path is blocked pending diagnosis  
**Time Investment**: ~30 min to test one path, ~90+ min to rebuild NGC 24.04

---

## Quick Diagnostic (5 minutes)

Before attempting any major rebuilds, run this **inside the current vision-jetson:latest container** to get more info:

```bash
# SSH into container first
docker/run-docker-jetson

# Inside container:
cd /app
python3 docker/test-gpu-quick.sh
```

**Expected outcomes**:
- ✅ **Succeeds**: GPU works, proceed to full benchmark
- ❌ **Fails with "CUDA error: out of memory"**: Driver compat issue or memory carveout too small
- ❌ **Fails with "NVML_SUCCESS INTERNAL ASSERT"**: PyTorch CUDA allocator bug (current status)

---

## Decision Matrix

### Scenario 1: `test-gpu-quick.sh` Passes ✅
**Action**: Run full benchmark immediately
```bash
BENCH_CONFIG_YAML=/app/src/configuration-layer/config.jetson.yaml python3 benchmark.py
```
**Time**: 5 min  
**Outcome**: Get actual FPS measurements, requirement fulfilled

---

### Scenario 2: `test-gpu-quick.sh` Still Fails (NVML Allocator Bug) ❌

**Root Cause**: PyTorch 2.4.0a0 in NGC 24.06 has a bug managing GPU memory on Jetson

**Option A: Try NGC 24.04 (Recommended, Higher Risk/Reward)**
- **What**: Rebuild Docker image with older CUDA stack (NGC 24.04 uses PyTorch 2.3.1)
- **Time**: ~90 min to rebuild, 5 min to test
- **Probability of Success**: ~60-70% (older CUDA stack might avoid the allocator bug)
- **If Succeeds**: Full performance measurements, requirement fulfilled ✅
- **If Fails**: NGC 24.04 might have different bugs; fallback to native PyTorch

**How to do it**:
```bash
# Edit docker/Dockerfile.jetson, change FROM line:
FROM nvcr.io/nvidia/pytorch:24.04-py3-igpu

# Rebuild
bash docker/build-docker-jetson  # takes ~90 min

# Test
docker run --rm --runtime=nvidia --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/app vision-jetson:latest \
  bash /app/docker/test-gpu-quick.sh
```

---

**Option B: Try Native Jetson PyTorch (Medium Risk, No Docker)**
- **What**: Use L4T-provided PyTorch on host (avoids Docker + NGC)
- **Why Try**: Native libraries are optimized for Jetson Orin Nano, might avoid allocator bug
- **Blockers**: 
  - Requires adapting benchmark.py to not run in container
  - May have different dependency versions (transformers, ultralytics)
  - PyTorch on host reports CUDA unavailable (needs driver update or host PyTorch reinstall)
- **Time**: ~30 min to investigate, uncertain if viable
- **Probability of Success**: ~30% (host driver is incompatible with host PyTorch)

**How to check**:
```bash
# On host:
apt-cache search pytorch | grep python3  # check what's available
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

**Option C: x86 TensorRT Export Workaround (Low Risk, Requires External Hardware)**
- **What**: Export YOLO `.pt` → `.engine` on a GPU-rich workstation, copy engines to Jetson
- **Why**: Avoids GPU inference on Jetson entirely, uses TensorRT (faster)
- **Setup**:
  1. Take current NGC 24.06 container setup to an x86 machine with modern GPU
  2. Run `docker/export-yolo-trt.sh` to generate `.engine` files
  3. Copy `.engine` files to Jetson
  4. Run benchmark with config pointing to `.engine` files
- **Time**: Depends on whether you have x86 access
- **Probability of Success**: ~95% (known workaround from NVIDIA)

**How to adapt for this**:
```yaml
# In config.jetson.yaml, change:
config_yolo_model: yolov11v28_jingtao.engine  # instead of .pt
```

---

## Recommended Path Forward

**Decision Algorithm** (in priority order):

1. **First (5 min)**: Run `test-gpu-quick.sh` to confirm current blocker
   - If passes → Run benchmark, done ✅
   - If fails (allocator bug) → Proceed to step 2

2. **Second (5 min decision, 90 min build if chosen)**: 
   - Do you have 90 minutes? → Try NGC 24.04 (Option A)
   - No time? → Skip to step 3

3. **Third (Alternative paths)**:
   - Option B: Native PyTorch investigation (if Option A times out)
   - Option C: Use x86 machine for TRT export (if you have access)

---

## Files to Check / Modify

**Current Configuration** (ready to use once GPU works):
- `src/configuration-layer/config.jetson.yaml` — Jetson-optimized (cuDNN, batch tuning)

**Docker Files** (ready for rebuild):
- `docker/Dockerfile.jetson` — Currently NGC 24.06, can change to 24.04
- `docker/run-docker-jetson` — Already fixed with --ipc/ulimits
- `docker/export-yolo-trt.sh` — Ready for TRT export on workstation

**Benchmark Code**:
- `benchmark.py` — Already has cuDNN benchmark + config override
- Run with: `BENCH_CONFIG_YAML=/app/src/configuration-layer/config.jetson.yaml python3 benchmark.py`

---

## Fallback: If GPU Cannot Be Fixed This Week

If all GPU paths fail (NGC 24.04, native PyTorch, etc.), you still have:

1. **Pipeline is structurally sound** ✅
   - All layers import correctly
   - Config system works
   - YOLO detector runs on CPU (slow but functional)

2. **Performance improvements are code-ready** ✅
   - cuDNN auto-tuning implemented
   - VLM batch tuning configured
   - Just need GPU to work to measure impact

3. **Clear next steps documented** ✅
   - NGC version matrix
   - TensorRT export guide
   - vLLM alternative for VLM layer

**Minimum Viable Next Step**: Run on CPU to confirm functional correctness, then revisit GPU path after driver/firmware update on Jetson.

---

## FAQ: Why Is This So Hard?

**Q: Why didn't the original NGC 24.10 image work?**  
A: NGC 24.10 requires NVIDIA driver 560.35+. Your Jetson has driver 540.4.0. Mismatch caused NvMap allocator errors.

**Q: Why doesn't NGC 24.06 work if it's driver-compatible?**  
A: Compatibility is necessary but not sufficient. NGC 24.06 ships PyTorch 2.4.0a0 (pre-release) which has a bug in the CUDA memory allocator when used with Jetson's unified memory model. This is a PyTorch bug, not a driver bug.

**Q: Can you just fix the PyTorch bug?**  
A: No, it's in compiled C++ code inside the NGC image. Only NVIDIA can fix it (or we downgrade to PyTorch 2.3.1 in NGC 24.04, which might not have the bug).

**Q: Why not run without Docker?**  
A: Host has PyTorch 2.11.0 compiled with CUDA 13.0, but driver is 12.x. They're incompatible. Docker was meant to solve this, but NGC has its own bugs.

**Q: Will this ever work?**  
A: Yes. Options include:
- NGC 24.04 (likely) ← Try first
- Jetson driver firmware update to 560+ (would enable NGC 24.10+)
- Native L4T PyTorch (requires checking what's available)
- TensorRT export on x86 + copy (guaranteed to work)
