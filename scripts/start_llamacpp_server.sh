#!/usr/bin/env bash
# start_llamacpp_server.sh — Launch llama-server with Gemma-4-E2B INT4 + vision encoder
#
# Run this BEFORE starting the pipeline with config.jetson.vlm-e2b-llamacpp.yaml.
# The server binds to localhost:8080 and exposes an OpenAI-compatible HTTP API.
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

echo "Starting llama-server (Gemma-4-E2B Q4_K_M + mmproj) on port ${PORT}..."
echo "Model:  ${MODEL}"
echo "Mmproj: ${MMPROJ}"
echo ""

exec "${LLAMA_SERVER}" \
    --model "${MODEL}" \
    --mmproj "${MMPROJ}" \
    --n-gpu-layers 99 \
    --ctx-size 2048 \
    --threads 6 \
    --threads-batch 6 \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    --mlock \
    --port "${PORT}" \
    --host 127.0.0.1 \
    --no-mmap
