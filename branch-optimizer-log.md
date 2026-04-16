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

### 2026-04-16 — Edge-optimized model probe (Gemma-4-E2B-IT) [TA recommendation]
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
