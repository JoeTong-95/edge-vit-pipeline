#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "== Device profile (Linux) =="
echo "Repo: $REPO_ROOT"
echo

echo "## OS / CPU / RAM"
uname -a || true
if command -v lscpu >/dev/null 2>&1; then lscpu | sed -n '1,25p'; fi
if command -v free  >/dev/null 2>&1; then free -h; fi
echo

echo "## GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found (OK if no NVIDIA GPU)."
fi
echo

echo "## Python env"
python3 -c "import sys; print('python:', sys.version)"
python3 -c "import cv2, numpy; print('cv2:', cv2.__version__); print('numpy:', numpy.__version__)" 2>/dev/null || true
python3 -c "import ultralytics; print('ultralytics: ok')" 2>/dev/null || true
python3 -c "import supervision; print('supervision: ok')" 2>/dev/null || true
python3 -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())" 2>/dev/null || true
python3 -c "import transformers; print('transformers:', transformers.__version__)" 2>/dev/null || true
echo

echo "## Pipeline device profile (video-only, includes VLM if enabled)"
echo "(Note: output includes layer-by-layer ROI/YOLO/VLM summary)"
python3 "src/evaluation-output-layer/benchmark.py"

