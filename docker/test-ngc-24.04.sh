#!/bin/bash
# Quick post-build GPU test script
# Run this once NGC 24.04 image build completes

set -e

echo "=== NGC 24.04 GPU Compatibility Test ==="
echo "Time: $(date)"
echo ""

# Test 1: Simple CUDA availability
echo "[TEST 1] CUDA availability in container"
docker run --rm --runtime=nvidia -v $(pwd):/app vision-jetson:latest python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024 // 1024} MB')
" 2>&1

echo ""
echo "[TEST 2] GPU allocation test"
docker run --rm --runtime=nvidia -v $(pwd):/app vision-jetson:latest python3 -c "
import torch
print('Testing 10 MB allocation...')
x = torch.zeros(2621440, dtype=torch.float32, device='cuda')
print('✓ Allocation succeeded')
del x
torch.cuda.empty_cache()
" 2>&1

echo ""
echo "[TEST 3] YOLO model load test"
bash /app/docker/test-gpu-quick.sh 2>&1

echo ""
echo "=== GPU Tests Complete ==="
