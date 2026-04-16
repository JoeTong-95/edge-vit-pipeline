#!/usr/bin/env bash
# start_llamacpp_server.sh — Launch llama-server with Gemma-4-E2B INT4 + vision encoder
#
# Run this BEFORE starting the pipeline with config.jetson.vlm-e2b-llamacpp.yaml.
# The server binds to localhost:8080 and exposes an OpenAI-compatible HTTP API.
#
# Jetson Orin Nano NvMap IOVMM constraint (empirically measured):
#   The NvMap pool holds ~1500 MiB effective for large cudaMalloc allocations.
#   Budget: 22 GPU text layers (866 MiB) + compute buffer (~520 MiB) = ~1386 MiB.
#   The mmproj vision encoder (532 MiB) is kept on CPU via LLAMA_ARG_MMPROJ_OFFLOAD=0
#   to free that budget for more text layers.
#   Using mmap (default, NOT --no-mmap) so text weights are one contiguous allocation.
#
# Requirements:
#   - llama.cpp built with CUDA: /home/jetson/llama.cpp/build/bin/llama-server
#   - Model:    src/vlm-layer/gemma-4-e2b-gguf/gemma-4-E2B-it-Q4_K_M.gguf
#   - MMproj:   src/vlm-layer/gemma-4-e2b-gguf/mmproj-gemma-4-E2B-it-Q8_0.gguf

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_SERVER="/home/jetson/llama.cpp/build/bin/llama-server"
MODEL="${REPO_ROOT}/src/vlm-layer/gemma-4-e2b-gguf/gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ="${REPO_ROOT}/src/vlm-layer/gemma-4-e2b-gguf/mmproj-gemma-4-E2B-it-Q8_0.gguf"
PORT=8080

if [[ ! -f "${LLAMA_SERVER}" ]]; then
    echo "ERROR: llama-server binary not found at ${LLAMA_SERVER}"
    echo "Build llama.cpp first:"
    echo "  cd /home/jetson/llama.cpp"
    echo "  PATH=/usr/local/cuda/bin:\$PATH cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87"
    echo "  PATH=/usr/local/cuda/bin:\$PATH cmake --build build --config Release -j\$(nproc)"
    exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
    echo "ERROR: Model file not found: ${MODEL}"
    exit 1
fi

if [[ ! -f "${MMPROJ}" ]]; then
    echo "ERROR: mmproj file not found: ${MMPROJ}"
    exit 1
fi

echo "Starting llama-server (Gemma-4-E2B Q4_K_M, 22/36 GPU layers, mmproj on CPU) on port ${PORT}..."
echo "Model:  ${MODEL}"
echo "Mmproj: ${MMPROJ} [CPU-only, NvMap budget constraint]"
echo ""

# LLAMA_ARG_MMPROJ_OFFLOAD=0 keeps the vision encoder (mmproj) on CPU so the
# NvMap IOVMM budget (~1500 MiB) is reserved for text-model weights + compute buffers.
exec env LLAMA_ARG_MMPROJ_OFFLOAD=0 "${LLAMA_SERVER}" \
    --model "${MODEL}" \
    --mmproj "${MMPROJ}" \
    --n-gpu-layers 22 \
    --ctx-size 2048 \
    --threads 6 \
    --threads-batch 6 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --port "${PORT}" \
    --host 127.0.0.1
