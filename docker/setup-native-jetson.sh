#!/usr/bin/env bash
# setup-native-jetson.sh
#
# Installs all pipeline dependencies for native (no-Docker) Jetson execution.
# Must be run AFTER JetPack 6.x is installed (CUDA 12.6, cuDNN 9, driver 540.x).
#
# Key constraint: PyTorch and torchvision must come from the Jetson AI Lab index
# (pypi.jetson-ai-lab.io/jp6/cu126) — standard PyPI wheels are compiled for
# CUDA 13.0 and will report "CUDA available: False" on this system.
#
# Usage:
#   bash docker/setup-native-jetson.sh

set -euo pipefail

TORCH_WHEEL="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl"
TORCHVISION_WHEEL="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl"

echo "==> Step 1: Installing pipeline deps (excluding torch/torchvision)"
pip3 install \
  "ultralytics>=8.3.0,<9.0.0" \
  "transformers>=5.0.0,<6.0.0" \
  "accelerate>=0.34.0,<1.0.0" \
  "huggingface_hub>=1.3.0,<2.0.0" \
  "safetensors>=0.4.5" \
  "sentencepiece>=0.2.0" \
  "timm>=1.0.9" \
  "einops>=0.8.0" \
  "numpy==1.26.4" \
  "matplotlib==3.8.4" \
  "pillow==10.3.0" \
  "supervision>=0.18.0" \
  "cffi>=1.15.1"

echo ""
echo "==> Step 2: Installing Jetson CUDA 12.6 torch 2.8.0 (overrides any PyPI torch)"
pip3 install --force-reinstall --no-deps "$TORCH_WHEEL"

echo ""
echo "==> Step 3: Installing compatible torchvision 0.23.0"
pip3 install --force-reinstall --no-deps "$TORCHVISION_WHEEL"

echo ""
echo "==> Verifying GPU access..."
python3 -c "
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    t = torch.zeros(1).cuda()
    print('GPU tensor allocation: OK')
else:
    print('ERROR: CUDA not available — torch wheel may be wrong version')
    import sys; sys.exit(1)
"

echo ""
echo "==> Setup complete. Run benchmark with:"
echo "    BENCH_CONFIG_YAML=src/configuration-layer/config.jetson.yaml python3 benchmark.py"
echo ""
echo "Note: PYTORCH_NO_CUDA_MEMORY_CACHING=1 is set automatically by benchmark.py"
echo "      to avoid the Jetson CUDACachingAllocator NvMap bug when loading the VLM."
