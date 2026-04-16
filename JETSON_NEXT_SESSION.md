# Jetson Optimization Session — Branch Ready for Next Steps

**Current branch**: `jetson-optimization-A` (5 commits, all Phase 1 + perf tuning code in place)

## Status Summary
- ✅ Docker environment fixed (Phase 1 complete)
- ✅ Performance tuning code implemented (cuDNN, VLM batching)
- ❌ GPU inference blocked by CUDA allocator bug (PyTorch 2.4.0a0 + Jetson unified memory)
- 📊 No benchmark results yet (waiting on GPU unblock)

## Files in This Branch
```
docker/Dockerfile.jetson          # NGC 24.04 variant (experimental)
docker/run-docker-jetson          # Fixed with ipc/ulimits/device passthrough
docker/export-yolo-trt.sh         # TensorRT export (deferred due to OOM)
docker/test-gpu-quick.sh          # Fast GPU test (no warmup, single frame)
docker/build-docker-jetson        # Build script (unchanged)
benchmark.py                      # Added cuDNN benchmark + config override
src/configuration-layer/config.jetson.yaml  # Jetson-optimized config
JETSON_OPTIMIZATION.md            # Full technical report + recommendations
```

## Next Session: Quick Start (Pick One)

### Option A: Try NGC 24.04 (Recommended First)
```bash
cd /home/jetson/Desktop/edge-vit-pipeline
bash docker/build-docker-jetson  # Takes ~90 min
# After build completes:
docker run --rm --runtime=nvidia --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/app vision-jetson:latest \
  bash /app/docker/test-gpu-quick.sh
# If it works: run full benchmark
# If still fails: try Option B
```

### Option B: Explore vLLM or CPU-Only Pipeline
- vLLM for VLM (avoids transformers GPU issues)
- CPU-only YOLO (slower but stable)
- See JETSON_OPTIMIZATION.md for detailed recommendations

### Option C: Use x86 Workstation for TRT Export
- Export YOLO `.pt` → `.engine` on a GPU-rich machine
- Copy `.engine` files to Jetson
- Update config to use `.engine` instead of `.pt`
- Run benchmark

## Quick Reference: What to Check

1. **Read first**: `JETSON_OPTIMIZATION.md` (full technical report)
2. **Session history**: `chat-history/jetson-optimization-session.md` (if readable)
3. **Docker logs**: Check `chat-history/docker-rebuild-24.04.txt` if build is retried

## Known Issues & Workarounds

**Issue**: CUDA allocator crash on first YOLO inference  
**Root Cause**: PyTorch 2.4.0a0 + NGC 24.06 + Jetson Orin Nano unified memory  
**Workarounds**:
- NGC 24.04 (older CUDA, might avoid bug) ← **Try First**
- Native Jetson PyTorch (not in Docker) ← **If NGC 24.04 fails**
- x86 export + copy (avoids GPU inference on Jetson) ← **Fallback**

## Branch Ready For
- ✅ Merge (once GPU is working and benchmarked)
- ✅ Further experimentation (all code clean, committed)
- ✅ Handoff to next developer (full docs + git history)
