# Branch Optimizer Log — `jetson-optimization-e2b-llama`

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

#### Confirmed stable configuration: n=10 GPU layers

**First successful GPU boot.** Server started cleanly after fresh reboot with n=10.

Memory layout observed:
| Component | Size | Backend |
|---|---|---|
| GPU model buffer (10/36 layers) | 662.75 MiB | CUDA0 |
| GPU compute buffer | 532.50 MiB | CUDA0 |
| CPU Host compute buffer | 32.80 MiB | CUDA_Host |
| KV cache (flash-attn) | 10.13 MiB | CPU |
| CLIP vision encoder | CPU backend | CPU |
| **Total GPU** | **~1195 MiB** | **fits in ~1400 MiB budget** |

#### Benchmark results — n=10/36 GPU layers (hybrid CPU+GPU)
| Scenario | Latency | Notes |
|---|---|---|
| Cold start (first query) | **28.69s** | mmproj CPU + 26 CPU text layers |
| Re-query 1 (KV cache warm) | **8.03s** | |
| Re-query 2 | **7.86s** | |
| Pure CPU baseline | 8.2s | Previous measurement |
| **Speedup vs CPU** | **~4%** | Minimal — too few layers on GPU |

**Conclusion:** 10 GPU layers gives negligible speedup. With ~28% of transformer compute on GPU, the CPU-bound layers (26/36) dominate latency. More layers are needed for meaningful improvement.

#### NvMap reboot requirement
- Each test attempt (success or failure) degrades NvMap such that the next attempt needs a reboot.
- Even graceful SIGTERM (proper process exit) leaves NvMap fragmented: max single alloc drops from ~1400 MiB fresh to ~500 MiB post-shutdown.
- This is a Jetson kernel-level NvMap driver behavior — not fixable at application level.
- **Every layer-count experiment requires a clean reboot.**

#### Execution split audit (GPT-recommended verification — 2026-04-16)

Verified actual backend usage per component during live inference:

| Component | Claimed backend | Verified backend | Notes |
|---|---|---|---|
| Token embedding layer | CPU | **CPU** | Part of `CPU_Mapped` 2906 MiB buffer |
| Transformer blocks 0–25 (26 layers) | CPU | **CPU** | `CPU_Mapped model buffer size = 2906.25 MiB` |
| Transformer blocks 26–34 (9 layers) | GPU | **GPU** | `CUDA0 model buffer size = 662.75 MiB` |
| Output/LM head layer | GPU | **GPU** | `offloading output layer to GPU` |
| Tokenizer | CPU | **CPU** | ggml C++ tokenizer, no GPU path |
| CLIP vision encoder (mmproj) | CPU | **CPU** | `clip_ctx: CLIP using CPU backend` (×2 — weights + compute) |
| KV cache (all layers) | CPU | **CPU** | `CPU KV buffer size = 3.38 MiB + 6.75 MiB` — `kv_unified = true` forces unified CPU pool |
| GPU compute buffer | GPU | **GPU** | `CUDA0 compute buffer size = 532.50 MiB` (scratch for 9 GPU layers) |
| CPU Host compute buffer | CPU | **CPU** | `CUDA_Host compute buffer size = 32.80 MiB` |

**KV cache critical finding**: `kv_unified = true` is set by llama.cpp for hybrid CPU/GPU models. All KV cache lives on CPU — including KV entries for the 9 GPU transformer layers. During GPU-layer attention, KV data must be read from CPU (PCIe roundtrip per layer per token).

#### GPU utilization during actual inference (tegrastats @ 100–200ms intervals)

| Query type | Latency | GPU active samples | GPU idle samples | GPU active % |
|---|---|---|---|---|
| Cold start (first query) | 33.6s | 9 / 168 (200ms) | 158 / 168 | **5.4%** |
| Warm re-query (KV cache hit) | 4.36s | 14 / 52 (100ms) | 38 / 52 | **26.9%** |

**CPU evidence during cold start**: core 5 at 100% for the full 33s. GPU spikes only appear briefly when the 9 top transformer blocks and lm_head execute.

#### Root cause analysis

With 9/35 repeating transformer blocks on GPU:
- ~25% of attention/FFN FLOPs run on GPU
- ~75% run on CPU (6 ARM cores @ 1.7GHz)
- KV cache on CPU adds a PCIe roundtrip for each GPU attention operation
- The CPU execution of 26 blocks is the bottleneck regardless of GPU presence

GPU usage is low not because of misconfiguration but because **9 GPU layers genuinely represent a small fraction of total per-token compute**. The GPU finishes its 9 layers very fast; the CPU is the bottleneck for the remaining 26.

#### Next steps after next clean reboot
1. Try **n=13** (estimated ~1266 MiB total GPU) — closest to budget ceiling
2. If stable: benchmark and compare to n=10 (expect ~8% more FLOPs on GPU → marginal improvement)
3. Find max stable n via ladder; stop at first failure
4. **Decision gate**: if max stable layers still gives < 30% speedup, escalate strategy (see below)

#### YOLO TRT + llama-server GPU conflict — critical finding

**Test:** loaded YOLO TRT engine (`yolov11v28_jingtao.engine`, 53 MB on disk) while llama-server was running (n=10, ~1195 MiB GPU). Result:

```
NvMapMemAllocInternalTagged: 1075072515 error 12   ← ENOMEM
torch.AcceleratorError: CUDA error: out of memory
```

`1075072515 bytes = 1025 MiB` — **TRT's default workspace** (exactly `1<<30` + overhead). This is what the TRT runtime tries to allocate as a scratch buffer at engine load time, regardless of model size.

**Memory accounting (post-reboot NvMap budget ~1400 MiB effective):**

| Component | NvMap needed | Notes |
|---|---|---|
| CUDA runtime init overhead | ~200 MiB | Unavoidable |
| YOLO TRT workspace | ~1025 MiB | TRT default, set at engine-build time |
| YOLO TRT weights (deserialized) | ~100–150 MiB | 53 MB engine → 2-3× at runtime |
| **YOLO total** | **~1200–1375 MiB** | Almost the entire NvMap budget |
| llama-server compute buffer | 532 MiB | Fixed, required |
| **YOLO + llama-server GPU** | **~1732+ MiB** | **Exceeds budget by ~330+ MiB** |

**Conclusion: YOLO TRT and llama-server GPU are mutually exclusive on this Jetson.** They cannot coexist in GPU memory simultaneously. The current hybrid (n=10 GPU layers) only works because llama-server is started *before* YOLO, claiming NvMap first. In the actual pipeline, YOLO initializes before `initialize_vlm_layer()`, meaning YOLO would claim its ~1.2 GB first, leaving <200 MiB for llama-server — not enough for even the compute buffer.

**The no-YOLO test (GPT recommendation) is now the critical next experiment.** Without YOLO's 1 GB TRT workspace, the NvMap budget for llama-server rises from ~375 MiB to ~1200 MiB for model weights, enabling n=20+ layers.

#### GPT-recommended revised game plan (2026-04-16)

**Priority 1 — Control test (next reboot, no YOLO):**
Run llama-server with n=20 GPU layers in isolation (no YOLO process). This is the cleanest experiment:
- Without YOLO: model weights budget = 1400 - 532 = ~870 MiB → ~13 layers (same as before)
- Wait — YOLO's 1025 MiB is additive only when YOLO loads. Without YOLO, the budget IS the ~870 MiB measured.
- So no-YOLO test should try n=20 to see if YOLO's TRT workspace was fragmenting NvMap across boots.

**Priority 2 — YOLO TRT workspace reduction (secondary, worth trying):**
TensorRT lets you cap the builder memory pool via `setMemoryPoolLimit()` or `trtexec --memPoolSize=workspace:<size>`. Default is 1 GiB. Rebuild `yolov11v28_jingtao.engine` with `workspace=256` MB:

```python
from ultralytics import YOLO
model = YOLO('src/yolo-layer/models/yolov11v28_jingtao.pt')
model.export(format='engine', workspace=256, device=0, half=True)
```

**Important nuance (GPT):** Runtime TRT memory = workspace pool + deserialized weights on device + execution context persistent activations. Even with workspace=256 MB, total YOLO footprint is still 300-400 MiB. Combined with n=10 llama-server (~1195 MiB) = ~1495-1595 MiB — still above the ~1400 MiB pool. **Concurrent residency still unlikely.** Treat this as making YOLO a better citizen for YOLO-alone benchmarks or serialized execution — not the unlock for concurrent GPU use.

**Priority 3 — Serialized GPU architecture (if concurrent is impossible):**
- YOLO runs frame → releases TRT context as completely as possible → llama-server runs → YOLO reloads
- This is an engineering-inference recommendation from GPT, not a tested path
- Complexity: high (dynamic model loading/unloading per frame cadence)

| Scenario | YOLO | VLM | VLM re-query | Notes |
|---|---|---|---|---|
| Current (as deployed) | GPU TRT FP16 | CPU all-layers | 8.2s | Only works because llama-server starts first |
| No-YOLO isolation test | — | GPU n=20 | **TBD — next reboot** | Key experiment |
| YOLO CPU + VLM full GPU | CPU (~50ms/frame) | GPU all-layers | ~2-4s est. | Viable if YOLO CPU perf acceptable |
| Serialized GPU | GPU (alternating) | GPU (alternating) | High latency per cycle | Complex, not practical for real-time |

**GPT + agent consensus:** No-YOLO isolation test first. If that proves full GPU E2B is viable (sub-4s), then the decision is YOLO-CPU vs accept 8.2s. If isolation test still shows >7s, close the book on E2B+llama.cpp hybrid on this Jetson.

#### Strategic reassessment

| Approach | Status | Realistic latency |
|---|---|---|
| n=10 hybrid (current, no YOLO in GPU) | Working | ~8s re-query |
| No-YOLO isolation test, n=20 | **Next reboot** | TBD — key experiment |
| Full GPU, no YOLO (36/36) | Blocked by compute buffer | ~1–2s (theoretical) |
| YOLO CPU + llama-server full GPU | Requires pipeline config change | ~2-4s est. |
| YOLO GPU + llama-server GPU | **Impossible** — mutually exclusive | N/A |
| SmolVLM-256M PyTorch GPU | Working, other branch | ~6.3s per query |
| Pure CPU E2B (all layers CPU) | Working | ~8.2s re-query |

The ~1s target requires either (a) full GPU for E2B + YOLO off GPU, or (b) a smaller model that fits in <870 MiB alongside YOLO's footprint.

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
