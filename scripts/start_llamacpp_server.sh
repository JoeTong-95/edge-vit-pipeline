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

echo "Starting llama-server (Gemma-4-E2B Q4_K_M, 36/36 GPU layers) on port ${PORT}..."
echo "Model:  ${MODEL}"
echo "Mmproj: ${MMPROJ} [weights on CPU, compute on GPU via flash-attn]"
echo ""

# Memory strategy (empirically validated on Jetson Orin Nano, post-reboot):
#
#   GGML_CUDA_ENABLE_UNIFIED_MEMORY=1
#     Switches GPU buffer allocation from cudaMalloc → cudaMallocManaged.
#     cudaMallocManaged uses non-contiguous physical pages (bypasses NvMap IOVMM
#     contiguous block limit), allowing the full 4-6 GB unified memory pool.
#     Without this, a single 544 MiB cudaMalloc for the compute buffer fails after
#     1416 MiB of model weights are loaded (NvMap contiguous limit ~1.9 GB).
#
#   --no-mmap
#     Forces each tensor to be allocated individually via ggml_cuda_device_malloc
#     (which respects GGML_CUDA_ENABLE_UNIFIED_MEMORY).  With mmap (default) the
#     GPU weight buffer is allocated as ONE large contiguous cudaMalloc block,
#     bypassing the unified-memory path and hitting the NvMap contiguous limit.
#
#   --no-mmproj-offload
#     Keeps the CLIP/mmproj vision-encoder weights on CPU.  The mmproj GPU weight
#     buffer (~216 MiB) would be the last allocation to fail even when everything
#     else fits, because it is a separate cudaMalloc outside the unified-memory
#     path.  Moving it to CPU avoids this.  Image encoding runs on CPU (6 threads),
#     text generation on full GPU (36/36 layers).
#
#   --flash-attn on
#     Reduces KV-cache from ~100 MiB → ~3 MiB (2048 ctx) and slightly shrinks the
#     compute-buffer footprint (544 → 538 MiB), giving the allocation just enough
#     headroom to succeed.

exec env GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 "${LLAMA_SERVER}" \
    --model "${MODEL}" \
    --mmproj "${MMPROJ}" \
    --n-gpu-layers 99 \
    --ctx-size 2048 \
    --flash-attn on \
    --no-mmproj-offload \
    --no-mmap \
    --threads 6 \
    --threads-batch 6 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --port "${PORT}" \
    --host 127.0.0.1
