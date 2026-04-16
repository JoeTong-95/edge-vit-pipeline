# GPU Memory Access Blocker - Deep Diagnostic (NGC 24.06)

**Date**: 2026-04-16, 00:25 AM  
**Status**: Critical blocker identified - even 4 MB allocations fail  
**Implication**: Not a PyTorch allocator bug specifically, but lower-level NvMap issue

---

## Key Finding

```
NvMapMemAllocInternalTagged: 1075072515 error 12
NvMapMemHandleAlloc: error 0
RuntimeError: CUDA error: out of memory
```

**This happens when**:
- Attempting ANY GPU tensor allocation, even 4 MB (trivial size)
- On the **first allocation** in a fresh Python process
- Inside the vision-jetson:latest container

**Error 12** = `ENOMEM` (no memory available) from the kernel NvMap driver

---

## Root Cause Analysis

This is **NOT** a PyTorch-specific bug. The issue is at the **kernel driver level**:

1. **NvMap** is a Jetson memory management driver that manages the GPU memory carveout
2. Error 12 means NvMap's carveout is exhausted or misconfigured
3. NGC 24.06 container is requesting GPU memory, but the kernel driver is rejecting it

**Possible causes**:
- NvMap carveout is too small (configured in L4T firmware)
- Container lacks permission to access NvMap nodes
- NGC 24.06 PyTorch is compiled with different memory alignment than L4T R36.4.7 supports
- Container needs kernel-specific tuning (not just --ipc-host/ulimits)

---

## Why NGC 24.04 Might Not Help

If the problem is **NvMap/kernel-level**, then:
- ✅ NGC 24.04 might help if it has different memory alignment expectations
- ❌ NGC 24.04 might also fail if it hits the same kernel limitation
- ✗ Pure version downgrade won't fix kernel-level incompatibility

**Implication**: Rebuilding NGC 24.04 is a **gamble**, not a guaranteed fix.

---

## More Likely Root Cause

The real issue might be that **L4T R36.4.7 is too old for any modern NGC image**.

Evidence:
- Jetson Orin Nano is from 2023
- L4T R36.x was released ~mid 2024
- NGC 24.06 was released ~June 2024
- They should be compatible, but edge cases happen

---

## Actual Recommended Actions (Revised)

Given this diagnostic, the priority order changes:

### Option 1: Upgrade L4T Firmware (5-30 min)
```bash
# Check current L4T version
cat /etc/nv_tegra_release

# If R36.4.7 or older, check for updates
apt list --upgradable | grep l4t

# If update available:
apt update && apt upgrade -y
reboot
```

**Why**: Newer L4T firmware (R36.5+) might have NvMap fixes or different behavior  
**Risk**: Low (should be backward compatible)  
**Time**: 5 min to check, 20-30 min if update available

### Option 2: NGC 24.04 Rebuild (90 min, Conditional)
Only do this **after** confirming L4T is fully updated.  
If L4T update doesn't help, NGC 24.04 might work differently with new L4T.

### Option 3: Use Native L4T PyTorch (10-30 min)
```bash
# Check what Jetson PyTorch is available
apt-cache search pytorch

# Install if available
apt install python3-pytorch

# Test GPU access natively (no Docker)
python3 benchmark.py
```

**Why**: Native PyTorch is compiled for L4T and likely avoids Docker/NvMap conflicts  
**Risk**: Might have version mismatches with other dependencies  
**Time**: 10 min to check, 30 min to test if available

### Option 4: TensorRT Export on x86 (Highest Success Probability)
Still recommended as fallback.

---

## Next Session Priority

1. **First** (5 min): Check L4T firmware version
   ```bash
   cat /etc/nv_tegra_release
   ```
   
2. **Second** (If update available, ~20-30 min): Update L4T
   ```bash
   apt update && apt upgrade
   reboot
   ```

3. **Third** (If after reboot, still need GPU, 10 min): Try native PyTorch
   ```bash
   apt-cache search pytorch
   ```

4. **Fourth** (If all else fails, 90 min): NGC 24.04 rebuild

---

## Conclusion

The GPU memory issue is **not a container/config problem** but rather a **kernel-level GPU memory access problem**. This changes the strategy:

- ✗ More Docker tuning won't help
- ✗ Smaller batches won't help (even 4 MB fails)
- ✓ L4T firmware update might help significantly
- ✓ Native PyTorch might bypass the Docker/NvMap interaction issue
- ✓ TensorRT export (avoids GPU inference on Jetson) will definitely work

**Recommendation**: Before NGC 24.04 rebuild, try L4T firmware upgrade. That's likely where the real fix is.
