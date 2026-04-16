# Native PyTorch GPU Investigation (2026-04-16 23:50 PM)

**Result**: ❌ Native PyTorch also GPU-broken

## Diagnostic

```
Native PyTorch: 2.11.0+cu130
CUDA available: False
Error: "NVIDIA driver on your system is too old (found version 12060)"
```

## Interpretation

- Host has PyTorch compiled for CUDA 13.0
- Host driver is version 540.4.0 (= 12.060)
- Mismatch: PyTorch 2.11.0 requires CUDA 13+, but driver is CUDA 12
- **Result**: GPU is not accessible from native PyTorch either

## Conclusion

**NGC Docker is the only viable GPU path** for this Jetson + host driver combo.

Options now:
1. ✅ **NGC 24.04 rebuild** (90 min, 60-70% success rate)
   - Older CUDA stack may avoid allocator bug
   - Only remaining option before driver update

2. ❌ Native PyTorch (blocked by driver version)

3. ⏸️ Driver/firmware update (outside scope, would require reboot + L4T updates)

## Decision

**Proceeding with NGC 24.04 rebuild** as the only viable path to GPU inference within this session.

Build started: 23:50 PM
Expected completion: 01:20 AM (gives 20 min for testing + fallback)
