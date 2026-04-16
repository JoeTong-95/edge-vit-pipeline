# Jetson System Baseline (as of 2026-04-16 00:30 AM)

## L4T & JetPack Versions
```
L4T Release: R36 (release)
L4T Revision: 4.7
L4T Kernel: oot (out-of-tree)
JetPack Version: 6.2.1
NVIDIA Driver: 540.4.0
```

## Hardware
```
Device: NVIDIA Jetson Orin Nano (Engineering Reference Developer Kit Super)
GPU Memory: 7619 MiB (8 GB unified with system RAM)
```

## Update Status
- JetPack 6.2.1 already installed (no newer version available in apt)
- No L4T firmware updates available
- **Conclusion**: System is already at latest stable L4T R36.x

## Implications for GPU Memory Issue

Since L4T R36.4.7 is the latest in R36 series:
- ❌ L4T firmware update won't help (already current)
- ✓ NGC 24.04 retry still viable (might have different assumptions)
- ✓ Native L4T PyTorch worth checking
- ✓ TensorRT export on x86 remains fallback

## Next Session Actions (Prioritized)

1. **Try Native L4T PyTorch** (10 min) - Check if PyTorch from Jetson repos works
   ```bash
   apt-cache search pytorch | grep python
   apt install python3-pytorch  # if available
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **NGC 24.04 Rebuild** (90 min) - If native doesn't exist or fails
   - Edit: `FROM nvcr.io/nvidia/pytorch:24.04-py3-igpu`
   - Run: `bash docker/build-docker-jetson`
   - Test: `docker run ... bash /app/docker/test-gpu-quick.sh`

3. **x86 TensorRT Export** (if both fail) - Use workstation GPU
   - Export `.engine` files on x86
   - Copy to Jetson
   - Update config to use `.engine`

## Baseline Diagnostic Log

```
Container: vision-jetson:latest (NGC 24.06)
PyTorch in container: 2.4.0a0+f70bd71
CUDA in container: Available (reports 7619 MiB VRAM)
Test: Even 4 MB GPU tensor allocation fails

Error on allocation attempt:
  NvMapMemAllocInternalTagged: 1075072515 error 12
  RuntimeError: CUDA error: out of memory
```

This confirms the GPU is detected but kernel NvMap driver rejects memory allocations. All tested solutions are software-level (NGC version, PyTorch, etc.).
