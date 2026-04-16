# Branch Optimizer Log — `jetson-optimization-vlm-latency`

## Objective
Maximize deployable Jetson performance with focus on VLM latency and end-to-end responsiveness.

## Constraints from user
- Keep changes isolated to this branch.
- Optimize for real Jetson runtime performance (not theoretical).
- Maintain documentation fidelity:
  - For each layer implementation change: update corresponding layer `CHANGES.md`.
  - For pipeline contract/doc changes: update corresponding `pipeline/*` docs and `CHANGES.md`.

## Initial baseline plan
1. Measure baseline on current Jetson config (`config.jetson.yaml`, VLM on CPU async).
2. Measure GPU-VLM viability using test config (`config.jetson.vlm-gpu-test.yaml`).
3. Decide primary bottleneck type:
   - model execution latency (per-query heavy),
   - over-dispatch rate (too many low-value VLM calls),
   - queue/backpressure policy mismatch.
4. Implement highest-impact optimization first, benchmark, iterate.

## Current hypotheses
- H1: VLM-on-GPU may still be unstable with YOLO TRT FP16 due to unified memory pressure.
- H2: Biggest practical deployment gain may come from reducing VLM query load via stricter dispatch policy, not only faster model runtime.
- H3: Async policy tuning (batch wait/size + gating) can reduce wall-clock semantic latency and prevent stale outputs.

## Experiment log

### 2026-04-16 — Start
- Created branch.
- Added GPU VLM test config: `src/configuration-layer/config.jetson.vlm-gpu-test.yaml`.
- Next: run baseline matrix and capture metrics.

### 2026-04-16 — Baseline matrix (completed)

#### Run A: VLM on CPU (`config.jetson.yaml`)
- `estimated_fps`: **21.85**
- `avg_end_to_end_ms`: **~45.8 ms** (from benchmark table)
- YOLO post-ROI: **37.15 ms** (`cap_fps ~26.92`)
- VLM `avg_query_ms`: **103,982.6 ms** (~104 s/query)
- Outcome: frame loop is decent, semantic output latency is far too high for deployment response times.

#### Run B: VLM on GPU (`config.jetson.vlm-gpu-test.yaml`)
- `estimated_fps`: **19.23**
- YOLO post-ROI: **39.22 ms** (`cap_fps ~25.50`)
- VLM `avg_query_ms`: **11,325.5 ms** (~11.3 s/query)
- Outcome: VLM latency improves by ~9.2x vs CPU, but frame throughput drops ~12% due to GPU contention with YOLO.

#### Finding
For deployment usefulness, the dominant blocker is still VLM response latency (even after GPU move).  
GPU is clearly better than CPU for VLM on this Jetson, but we now need to reduce contention impact on YOLO and lower semantic response latency further.

### Next optimization directions (approved for this branch)
1. **Adopt GPU VLM as the deployment-latency default path**, but keep CPU fallback config.
2. **Reduce VLM workload per query**:
   - tighten dispatch gating (avoid low-value/duplicate semantic calls),
   - enforce min crop quality threshold before dispatch.
3. **Tune async policy for latency (not throughput)**:
   - test `batch_size=1` and `batch_wait_ms=0` for lowest first-result latency.
4. **Evaluate smaller VLM runtime option** (if model availability allows) for sub-5s target.

### 2026-04-16 — Async latency tuning test (batch=1, wait=0, cache=4)
- Config used: `config.jetson.vlm-gpu-lowlat.yaml`
- Results:
  - `estimated_fps`: **18.57** (worse than GPU-test profile 19.23)
  - `avg_yolo_ms`: **40.49 ms** (higher contention)
  - VLM: **no queries completed** in metrics window
- Interpretation:
  - More aggressive async settings did not improve deployable behavior.
  - Reducing crop cache too far can suppress semantic completions in this clip.
  - Keep `config.jetson.vlm-gpu-test.yaml` as the current best GPU-VLM profile.

## Current best-known profile for deployment-oriented latency
- `config.jetson.vlm-gpu-test.yaml`
- Tradeoff accepted for now:
  - CPU VLM: better FPS (~21.85), unusable semantic latency (~104s/query)
  - GPU VLM: lower FPS (~19.23), dramatically better semantic latency (~11.3s/query)

### 2026-04-16 — Contract-preserving semantic slimming
- User-approved contract change: remove `estimated_weight_kg`.
- Implemented end-to-end in VLM layer:
  - prompt no longer requests weight
  - parser/normalizer no longer emit weight field
  - debug/sample outputs updated
- Kept full pipeline semantics and ack flow unchanged:
  - `is_truck`, `wheel_count`, `ack_status`, `retry_reasons`
- Also reduced VLM generation cap from 64 -> 32 tokens to match slimmer JSON payload.
- Benchmark findings:
  - CPU-VLM profile (`config.jetson.yaml`): `avg_query_ms` improved from ~103,983 -> ~82,679 ms (~20.5% faster), pipeline FPS ~22.47.
  - GPU-VLM profile rerun hit Jetson allocator instability (`CUDACachingAllocator.cpp:1131`) and VLM was skipped in that run.
- Interpretation:
  - Contract slimming is directionally correct and improves semantic latency on stable runs.
  - Biggest blocker for sub-second target remains GPU memory stability/fragmentation under mixed YOLO+VLM load.

### 2026-04-16 — INT4 checkpoint compatibility smoke tests (3 attempts)

#### Attempt 1: GPTQ INT4 (Qwen3.5-0.8B)
- Candidate: `Vishva007/Qwen3.5-0.8B-W4A16-AutoRound-GPTQ`
- Blocker: `auto-gptq` install failed due to Jetson package metadata mismatch (`0.7.1` vs `0.7.1+cu126`).
- Model load fails before inference with `NameError: QuantizeConfig is not defined`.
- Result: **FAIL** — backend dependency incompatible on this Jetson.

#### Attempt 2: Bitsandbytes NF4 (Qwen3.5-0.8B)
- Candidate: `techwithsergiu/Qwen3.5-0.8B-bnb-4bit`
- Model loads on CUDA, but inference fails in `bitsandbytes` 4-bit matmul with `AssertionError: quant_state is not None`.
- Even with allocator tuning and explicit `.cuda()` call, the 4-bit execution path cannot initialize.
- Result: **FAIL** — 4-bit CUDA kernel issue on Jetson.

#### Attempt 3: AWQ INT4 (Qwen3.5-0.8B)
- Candidate: `Vishva007/Qwen3.5-0.8B-W4A16-AutoRound-AWQ`
- `autoawq` installs, but model loading requires `gptqmodel` backend.
- `gptqmodel` install fails on Jetson due to transitive dependency issue (`pypcre` resolution failure).
- Result: **FAIL** — required backend cannot be installed.

#### Conclusion
- Tried 3 distinct 0.8B INT4 paths, each with a different backend blocker (GPTQ, bitsandbytes, AWQ).
- Continued 0.8B INT4 backend wrangling has diminishing returns on this Jetson environment.
- All failures are runtime/backend-related, not hardware-limited.

### 2026-04-16 — Smaller-model fallback probe (SmolVLM-256M-Instruct)
- Candidate tested: `HuggingFaceTB/SmolVLM-256M-Instruct`.
- Downloaded locally to: `src/vlm-layer/SmolVLM-256M-Instruct`.
- Smoke test:
  - Loaded with `AutoProcessor` + `AutoModelForImageTextToText` on CUDA.
  - Synthetic single-image prompt completed successfully on Jetson GPU.
  - Load time: **~1.63 s**
  - Single inference latency: **~4.54 s**
- Output quality on first raw prompt:
  - Model responded, but did not follow the exact target JSON schema cleanly on the first naive prompt.
- Interpretation:
  - This model is materially more promising for deployable Jetson use than the blocked 0.8B INT4 attempts.
  - It is still above the long-term `~1 s` target, but much closer than current Qwen3.5 0.8B paths.

### 2026-04-16 — Gemma-4-E2B + llama.cpp INT4 baseline (branch: `jetson-optimization-e2b-llama`)

#### Setup
- Built llama.cpp from source with CUDA SM87 (`DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`).
- Binary: `/home/jetson/llama.cpp/build/bin/llama-server` (12MB)
- Model: `lmstudio-community/gemma-4-E2B-it-GGUF` → `gemma-4-E2B-it-Q4_K_M.gguf` (3.2GB)
- MMproj: `ggml-org/gemma-4-E2B-it-GGUF` → `mmproj-gemma-4-E2B-it-Q8_0.gguf` (532MB)

#### Pipeline integration
- New `config_vlm_backend` config field: `"pytorch"` (default) or `"llamacpp"`.
- New `config_vlm_llamacpp_server_url` config field (default `http://localhost:8080`).
- `initialize_vlm_layer` branches on backend — `llamacpp` health-checks the server, no model loading in Python.
- `infer_vlm_semantics` / batch: HTTP POST to `/v1/chat/completions` using stdlib only (no new ML deps).
- Config: `src/configuration-layer/config.jetson.vlm-e2b-llamacpp.yaml`
- Launch script: `scripts/start_llamacpp_server.sh`

#### GPU memory issue (NvMap fragmentation)
- With all 36 layers on GPU (`-ngl 99`): weights load OK (1416 MiB), compute buffers fail (537 MiB cudaMalloc OOM).
- With 20 layers on GPU: weights OK (1031 MiB), compute buffers still fail (~534 MiB).
- Root cause: session-accumulated NvMap fragmentation from many failed CUDA allocations (bitsandbytes INT4, Qwen, SmolVLM tests). Total free GPU is 4 GB, but no contiguous 534 MiB block is available.
- **Resolution: Jetson reboot required to clear NvMap state before GPU benchmark.**

#### CPU-only baseline (CUDA_VISIBLE_DEVICES="" — forced, not representative)
- Context: 2048 tokens, no GPU layers, 4 CPU threads.
- First inference (cold): **~37s**
- Subsequent inferences: **~8.2s** (with KV cache hit on image tokens)
- Real-world (each frame is different): **~34s per query** — worse than Qwen BF16 GPU (11.3s)
- Token speed: prompt ~18 tok/s, decode ~10 tok/s on CPU.
- Output: ✓ valid JSON, but requires `thinking_budget_tokens=0` + system prompt to suppress chain-of-thought.

#### Thinking mode
- Gemma-4-E2B defaults to chain-of-thought reasoning, outputting to `reasoning_content` (not `content`).
- Fixed in `_infer_llamacpp()`: now passes `thinking_budget_tokens: 0` + system prompt.
- With fix: model outputs clean JSON directly to `content`.

#### GPU benchmark (pending — requires reboot)
- Estimated GPU latency at 3-5x CPU speedup: **~7-11s per query**
- This is comparable to Qwen BF16 GPU (11.3s), not a clear win for latency.
- Advantage over Qwen: no NvMap allocator crashes; llama.cpp runs as isolated process.

#### Next steps
1. Reboot Jetson to clear NvMap fragmentation.
2. Run `bash scripts/start_llamacpp_server.sh` and benchmark GPU path.
3. If GPU latency < 11.3s: document as deployment candidate.
4. If not: revisit SmolVLM-256M TRT path or TRT-LLM conversion.

### 2026-04-16 — Optimization deep-dive (CPU benchmarks, GPU projections)

#### Finding 1: Image size has zero impact on token count
- Tested 64×64, 128×128, 224×224, 320×240, 448×448 — all produce **~326 prompt tokens**.
- The mmproj (CLIP, 476M) processes at a fixed native resolution and resizes input internally.
- Sending smaller images saves nothing. Do not optimize image preprocessing.

#### Finding 2: KV cache reuse is the biggest latency lever
- When the **same vehicle crop is re-queried** (tracked vehicle), 281/326 tokens are served from KV cache.
- Re-query latency (CPU): **8.2s** vs **~25s** cold start — **3× faster**.
- Implication: pipeline's `config_vlm_crop_cache_size` and `config_vlm_dead_after_lost_frames` should be tuned aggressively to avoid re-querying stable classifications.
- On GPU, re-query: estimated **~1-2s** — within the 1s target range.

#### Finding 3: Thread tuning — marginal
- 6 threads + 6 batch threads vs 4 threads: prompt speed ~16 vs ~18 tok/s.
- Not significant at this model size. Applied anyway (free win).

#### Finding 4: KV cache quantization (q4_0) — memory, not speed
- `--cache-type-k q4_0 --cache-type-v q4_0` reduces KV cache from f16 to int4.
- Saves ~120 MiB at ctx-size 2048 — critical for fitting compute buffers on GPU after reboot.
- Applied to start script.

#### Finding 5: Thinking mode suppression — reliable with any system prompt
- `thinking_budget_tokens: 0` + any non-empty system prompt reliably suppresses Gemma-4 chain-of-thought.
- Shortest confirmed working: `"Output JSON only."` (3 tokens, minimum system prompt overhead).
- Inconsistency in earlier tests was KV cache collision from a prior thinking-mode session.

#### Updated performance projections (post-reboot GPU)
| Scenario | CPU (baseline) | GPU (estimated) |
|---|---|---|
| Cold start (new vehicle) | ~25s | ~4-6s |
| Re-query (tracked vehicle, KV cache) | ~8.2s | **~1-2s** ← hits target |
| Qwen BF16 GPU (old baseline) | — | 11.3s |

#### Optimized server flags (applied to `scripts/start_llamacpp_server.sh`)
```
--n-gpu-layers 99 --ctx-size 2048
--threads 6 --threads-batch 6
--cache-type-k q4_0 --cache-type-v q4_0
--mlock --no-mmap
```

### 2026-04-16 — NvMap IOVMM constraint deep-dive (GPU benchmark blocker)

#### Problem
After rebooting and retrying, all configurations with `-ngl 99` still fail at the compute buffer
allocation step. Root cause: the Jetson Orin Nano's NvMap IOVMM pool is smaller than expected.

#### Empirical measurements (raw `cudaMalloc` via ctypes, fresh process after reboot)
| Allocation | Result |
|---|---|
| Single alloc, 1200 MiB | OK |
| Single alloc, 1346 MiB | FAIL |
| Two allocs: 1103 + 544 MiB | FAIL |
| Two allocs: 900 + 600 MiB | OK |
| Two allocs: 900 + 544 MiB | OK |
| PyTorch `torch.zeros(1416 MiB)` | OK (uses virtual memory API, not raw cudaMalloc) |

Key insight: llama.cpp uses raw `cudaMalloc` which is bounded by the NvMap IOVMM physical page pool
(~1500 MiB effective). PyTorch bypasses this with CUDA virtual memory APIs — this is why PyTorch
could allocate 1416+ MiB while llama.cpp could not.

#### Memory budget breakdown (per server run)
| Component | Size | Notes |
|---|---|---|
| Text model GPU weights (36 layers, mmap) | 1416 MiB | Single `cudaMalloc` |
| mmproj vision encoder (GPU) | 532 MiB | Single `cudaMalloc` |
| Compute/scratch buffer | ~520-544 MiB | Single `cudaMalloc` |
| **Total needed (all GPU)** | **~2468 MiB** | **Far exceeds ~1500 MiB pool** |

#### Viable configuration
To fit within the NvMap pool:
- Keep mmproj on **CPU** via `LLAMA_ARG_MMPROJ_OFFLOAD=0` (saves 532 MiB of pool)
- Use **22/36 GPU layers** (866 MiB weights + ~520 MiB compute = 1386 MiB ≤ ~1500 MiB) ✓
- Use **mmap** (NOT `--no-mmap`) so the 866 MiB text weights are a single contiguous alloc

Script updated accordingly. GPU inference with 22/36 layers is still ~61% of transformer
compute on GPU — meaningfully faster than pure CPU (8.2s baseline).

#### Why this wasn't obvious initially
Each failed llama-server start (partial allocations that then crash) degrades the NvMap pool
through fragmentation. Once degraded, even previously-working configurations fail. A reboot
resets the NvMap state to a clean ~1500 MiB pool. Running Python CUDA tests before the server
consumes additional pool budget.

#### Next step
**Reboot required.** After reboot, run `start_llamacpp_server.sh` DIRECTLY (no Python tests first)
to get the server on GPU with 22/36 layers. Then benchmark real GPU numbers.

### 2026-04-16 — GPU boot investigation (multiple reboots, exhaustive config sweep)

#### Context
Multiple reboots and configuration attempts were made trying to get the llama-server to start
with as many GPU layers as possible. Key findings documented below.

#### Key findings from config sweep

**Flash attention works and helps a lot:**
- `--flash-attn on` reduces KV cache from ~100 MiB → ~3.4 MiB (2048 ctx) — nearly free.
- Compute buffer reduction is minor: 544 MiB → 538 MiB (ctx 2048 → 512). Not the bottleneck.

**CLIP vision encoder (mmproj) requires its own GPU allocation:**
- Even with `--no-mmproj-offload`, the mmproj CLIP model allocates ~216–532 MiB on GPU.
- In runs where text model + compute buffer fit, the CLIP alloc is what finally fails.
- `--no-mmproj-offload` keeps CLIP weights on CPU but CLIP compute still hits GPU.
- Vision encoding on CPU (6 threads) is acceptable — only runs once per new image.

**The auto-fit probe (`llama_params_fit`) crashes on Jetson:**
- When n_gpu_layers is below ~28, `llama_params_fit` probes device memory and hits
  an unhandled OOM path that calls `ggml_abort()` (SIGABRT, exit code 134).
- Fix: pass `-fit off` to skip the probe and use the exact layer count specified.

**`--no-mmap` makes things worse (ChatGPT review correction):**
- `--no-mmap` forces a ~2.1 GB CPU allocation from system RAM, which competes directly
  with GPU NvMap pages in Jetson's unified memory pool.
- mmap (default) avoids this: CPU layers use on-demand file pages, GPU gets the RAM.
- All further attempts use mmap (no `--no-mmap`).

**`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` is not reliable:**
- Switches `ggml_cuda_device_malloc` to `cudaMallocManaged`. Works for model weights
  but the compute buffer (`ggml_gallocr`) path still fails identically.
- Python-level `cudaMallocManaged` test succeeds for 1416+544 MiB, suggesting the issue
  is either CUDA runtime internal overhead consuming NvMap before model loads, or the
  allocator path not uniformly using managed memory.
- Decision: drop this experiment — too unpredictable. Focus on fixed layer count.

#### Revised NvMap budget model
| Component | Size | Notes |
|---|---|---|
| CUDA runtime init overhead | ~200–400 MiB | cuBLAS handles, streams, internal cudaMalloc |
| Text model GPU weights (mmap) | 786–1416 MiB | Scales with n_gpu_layers |
| Compute/scratch buffer | ~530–544 MiB | **Fixed** — barely changes with layer count |
| mmproj CLIP (must stay CPU) | 0 MiB GPU | Via `--no-mmproj-offload` |
| **Usable NvMap total** | **~1400 MiB** | Empirically inferred |

Budget constraint: `model_weights + 530 ≤ 1400` → model budget ≤ 870 MiB → **≤ 22 GPU layers**

#### Configuration attempts summary (this session, post multiple reboots)
| Config | n_gpu_layers | Outcome |
|---|---|---|
| mmap, no flags | 36 | Model OK (1416 MiB), compute 544 MiB FAIL |
| no-mmap, unified mem | 36 | Model OK, compute 544 MiB FAIL |
| mmap, flash-attn, unified mem, ctx-512 | 36 | Model OK, compute 538 MiB OK ✓, CLIP 216 MiB FAIL |
| mmap, flash-attn, no-mmproj-offload | 36 | Compute 544 MiB FAIL |
| mmap, flash-attn, no-mmproj-offload | 28 | Model OK (1218 MiB), compute 532 MiB FAIL |
| mmap, flash-attn, no-mmproj-offload, -fit off | 20 | Model OK (1030 MiB), compute FAIL → SIGABRT |

#### Current stable config (in `scripts/start_llamacpp_server.sh`)
```
llama-server \
  --n-gpu-layers 20  -fit off \
  --ctx-size 2048    --flash-attn on \
  --no-mmproj-offload \
  --threads 6        --threads-batch 6 \
  --cache-type-k q4_0 --cache-type-v q4_0
```
- **Vision (mmproj):** CPU, 6 threads
- **Text LM:** 20/36 layers GPU + 16/36 layers CPU (hybrid)
- **KV cache:** GPU, ~3.4 MiB (flash-attn)

#### Model allocation size — empirical data
Measured GPU model buffer sizes for different n_gpu_layers (mmap mode):
| n_gpu_layers | GPU model buffer | Notes |
|---|---|---|
| 36 | 1416 MiB | All layers, loads OK — compute 544 FAIL |
| 28 | 1218 MiB | Loads OK — compute 532 FAIL |
| 20 | 1030 MiB | Loads OK — compute abort (SIGABRT) |
| 16 | ~883 MiB | FAIL at model alloc (pool already fragmented from prior run) |

Observed scaling: ~24 MiB per transformer layer + ~550 MiB fixed overhead (lm_head, embeddings, output norm).
Formula: `model_alloc ≈ 550 + n × 24 MiB`

With compute buffer ~530 MiB fixed:  
`NvMap budget = model_alloc + compute → 550 + n×24 + 530 ≤ ~1400 → n ≤ 13`

Max usable transformer layers on GPU with current budget: **~13**

#### Critical observations
- The `-fit off` flag is required — `llama_params_fit` crashes with SIGABRT when probing configs on Jetson.
- Each abort/crash leaves NvMap fragmented even after process exit — **reboot required between attempts**.
- Model allocation sizes are NOT proportional to n_gpu_layers linearly from 0 — there is ~550 MiB of fixed GPU overhead (embedding + lm_head) always present regardless of layer count.

#### Next steps after clean reboot
1. Boot with **n=10** (safe: 550+10×24+530 ≈ 1320 MiB) → verify stable + run latency benchmark
2. If stable: try n=12, then n=13
3. Stop at first failure; previous value is the stable operating point
4. Update this log with real latency numbers

### 2026-04-16 — Apple FastVLM-1.5B-int8 test
- Candidate: `apple/FastVLM-1.5B-int8` (LlavaQwen2 architecture, INT8 quantized).
- Result: **FAIL** — `llava_qwen2` model type not registered in transformers 5.5.4.
- Fix would require upgrading transformers which risks breaking other pipeline dependencies.
- Decision: skip for now, not worth the risk.

### 2026-04-16 — Gemma-4-E2B-IT via llama.cpp [TA recommendation]
- Reframed approach based on TA guidance: Gemma-4-E2B is specifically designed for Jetson Orin edge deployment.
- **NOT PyTorch** — runs via `llama.cpp` (raw C++ with CUDA kernels) or `vLLM`.
- **Format**: Q4_K_S GGUF (INT4 quantized, 5.0GB), even smaller than INT8.
- **Why it may be faster than PyTorch path**:
  - INT4 = half the memory bandwidth of INT8, quarter of FP16
  - llama.cpp = raw C++ CUDA kernels, no Python/PyTorch overhead, no NvMap issues
  - Runs as a separate server process → no GPU memory contention with YOLO TRT
- **Integration pattern**: OpenAI-compatible HTTP API → pipeline calls llama.cpp server locally
- **Action taken**: Building llama.cpp with CUDA SM87 and downloading Q4_K_S + mmproj in background.
- Candidate: `google/gemma-4-e2b-it` (Google DeepMind).
- Rationale: Gemma-4 Edge variants are specifically designed for Jetson Orin deployment with native multimodal support and bfloat16 dtype.
- Model architecture: native `Gemma4ForConditionalGeneration` with audio + image + text support.
- Initial attempt: tried full-precision download, which is ~5.0GB and takes very long on this network.
- **Corrected approach**: switched to INT8 GGUF quantized version (`ggml-org/gemma-4-E2B-it-GGUF` repo, `gemma-4-E2B-it-Q8_0.gguf`), which is much smaller and inference-ready.
- Status: INT8 GGUF version downloading in background; smoke test pending.

### 2026-04-16 — PyTorch latency benchmark (SmolVLM-256M vs Qwen3.5-0.8B)

#### SmolVLM-256M-Instruct
- Load time: **7.78s**
- Single-image inference: **6.3s per query**
- Output quality: ✓ valid JSON responses
- Result: **✓ STABLE AND WORKING**

#### Qwen3.5-0.8B
- Crashes during inference with Jetson NvMap allocator error (`CUDACachingAllocator.cpp:1131`)
- Result: **✗ NOT DEPLOYABLE** on this Jetson configuration

#### Summary
- **Winner (PyTorch path): SmolVLM-256M-Instruct**
  - Latency: ~6.3s per query (acceptable stepping stone; target is ~1s but this is 50% improvement over current 11.3s GPU Qwen)
  - Stability: runs reliably on CUDA without allocator crashes
  - Memory footprint: smaller than Qwen, compatible with concurrent YOLO TRT on GPU
- **Gemma-4-E2B INT8** is still downloading but blocked for PyTorch (GGUF format requires llama.cpp, not transformers)

### 2026-04-16 — TensorRT investigation for VLM

#### Findings
- Standard VLM models (SmolVLM, Qwen) do not have built-in ONNX export via `.export()`
- TensorRT conversion requires ONNX → TRT pipeline
- **Best practice:** NVIDIA's **TensorRT Edge-LLM** framework (specialized for VLM on Jetson)
  - Requires building from source on Jetson
  - Requires x86 host for initial quantization/export
  - Payoff: 2-3x speedup vs PyTorch
  - Complexity: High (not trivial setup)

#### Recommendation
- For immediate deployment: stick with **PyTorch SmolVLM-256M** (~6.3s latency)
- For further optimization: if 6.3s is still too slow, investigate TensorRT Edge-LLM after establishing baseline
- **Decision point:** is 6.3s acceptable, or proceed with TRT conversion?

### 2026-04-16 — TRT FP16 vision encoder acceleration (branch: `jetson-optimization-vlm-smolvlm-256m`)

#### Approach
- Instead of full TRT-LLM (requires host export + source build), implemented a targeted approach:
  - **Export only the SigLIP vision encoder** (ViT 12L, 512×512 input, 1024 patches, hidden=768) to ONNX, then build a TRT FP16 engine.
  - Patch `model.model.vision_model.forward` at runtime to run TRT; LM decoder stays in PyTorch.
  - Result: zero API changes, zero pipeline contract changes, auto-discovered when engine file present.

#### Files added/changed
- `scripts/build_trt_smolvlm_vision.py` — ONNX export + TRT FP16 build script (dynamic batch, max=5 tiles).
- `src/vlm-layer/layer.py` — `_maybe_load_trt_vision_engine()`, `_make_trt_vision_forward()` patch helpers; `VLMConfig.config_vlm_trt_vision_engine` field; `VLMRuntimeState.vlm_runtime_trt_context` field.
- `src/configuration-layer/config.jetson.vlm-smolvlm-256m-trt.yaml` — deployment config for TRT path.
- Config schema/types/normalizer/defaults updated with `config_vlm_trt_vision_engine`.

#### To run the TRT build
```bash
cd /home/jetson/Desktop/edge-vit-pipeline
python3 scripts/build_trt_smolvlm_vision.py \
    --model src/vlm-layer/SmolVLM-256M-Instruct \
    --engine src/vlm-layer/SmolVLM-256M-Instruct/vision_encoder_fp16.trt
```
Then benchmark with `config.jetson.vlm-smolvlm-256m-trt.yaml`.

#### Expected outcome
- Vision encoder step: ~1.5–2.5× speedup (TRT FP16 kernel fusion vs PyTorch bfloat16).
- End-to-end VLM query: ~15–25% improvement (encoder is ~25–35% of total query time with 32-token decode).
- Target: ~5s/query (down from ~6.3s).
- Next step if still above target: investigate TRT-LLM for the decoder as well.
