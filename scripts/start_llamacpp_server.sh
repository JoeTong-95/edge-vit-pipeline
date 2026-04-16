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

# Stable hybrid config (empirically determined on Jetson Orin Nano):
#
# The NvMap IOVMM pool after CUDA runtime init has ~1400 MiB of usable contiguous
# space. The compute/scratch buffer is always ~530 MiB regardless of layer count.
# That leaves ~870 MiB for model weights = ~22 GPU layers.  We use 20 here as the
# conservative stable starting point.  Increment by 1–2 only after confirming
# repeatable boot + stable inference.
#
#   --n-gpu-layers 10   : ~790 MiB GPU weights + ~510 MiB compute ≈ 1300 MiB total
#   -fit off            : skip the auto-fit probe (crashes with abort() on Jetson)
#   --no-mmproj-offload : keep CLIP vision encoder on CPU (its GPU alloc fails too)
#   --flash-attn on     : KV cache 100 MiB → 3 MiB (huge reduction)
#   mmap (default)      : model loads with one GPU alloc, no 2 GB CPU RAM hit

exec "${LLAMA_SERVER}" \
    --model "${MODEL}" \
    --mmproj "${MMPROJ}" \
    --n-gpu-layers 10 \
    -fit off \
    --ctx-size 2048 \
    --flash-attn on \
    --no-mmproj-offload \
    --threads 6 \
    --threads-batch 6 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --port "${PORT}" \
    --host 127.0.0.1
