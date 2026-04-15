## VLM performance optimization note (2026-04-15)

### What’s the problem?

In the current stack, object-level VLM enrichment can fall behind when multiple tracks become dispatchable, or when per-query latency is high relative to how fast objects move through the frame.

Concrete issues observed in the architecture:

- **No true batching in the VLM layer**: the core inference path was effectively batch-size-1, paying processor + device-transfer + generate overhead per crop.
- **Backpressure only per-track**: the cropper prevents more than one in-flight request per track, but there is no global scheduler. If many tracks become dispatchable at once, realtime inference can lag.
- **Realtime vs “truck crosses the frame”**: if VLM latency is too large, a left→right truck can be gone before semantics return, even if YOLO+tracking kept up.

### What was attempted?

Two complementary approaches were pursued:

1) **Increase realtime throughput** (reduce per-crop amortized cost) by adding true **batched inference** and enabling **micro-batching** for queued VLM work.

2) Add an explicit **cache-and-run** escape hatch: when realtime cannot keep up, persist the best available crops + request metadata and run VLM offline later, instead of losing the evidence.

### What changed?

Changes landed on `test/all-except-camera` via commit `7de0ac0`:

- **`src/vlm-layer/layer.py`**
  - Added `infer_vlm_semantics_batch(...)` and `run_vlm_inference_batch(...)`.
  - Reduced `max_new_tokens` for the single-crop inference call to `64` (the contract is small JSON).

- **`src/vlm-layer/visualize_vlm_realtime.py`**
  - Upgraded `AsyncVLMWorker` to do **micro-batching**:
    - `--vlm-batch-size` (default `1`)
    - `--vlm-batch-wait-ms` (default `20`)
  - Added optional overflow **spill queue** (“cache-and-run”) if the worker queue is full:
    - `--vlm-spill-queue <path.jsonl>`
  - Spill behavior writes the crop + metadata to disk and emits an immediate **accepted** ack with reason `deferred_spill_to_queue` so the cropper does not stall on `vlm_request_in_flight`.

- **Deferred queue tooling**
  - `src/vlm-layer/vlm_deferred_queue.py`: JSONL format + PNG base64 encode/decode.
  - `src/vlm-layer/run_deferred_vlm_queue.py`: offline batch processor to compute semantics later.

- **Documentation**
  - `src/vlm-layer/VLM_OPTIMIZATION_NOTES.md`: rationale, usage examples, and guidance on realtime vs cache-and-run.

### What is it like now?

- **Realtime mode** can run VLM work in the background while keeping the frame loop moving, and can extract more throughput via micro-batching on GPU.
- **Overload behavior is controllable**:
  - If you want strict realtime, you can cap queue sizes and accept drops.
  - If you want to avoid losing transient events, you can spill to disk and process later.

Recommended first checks for a given device:

- Run `src/evaluation-output-layer/benchmark.py` and compare the **observed dispatch rate** vs **avg VLM ms/query**. You need roughly \( \lambda \cdot T < 1 \) to keep up without backlog.
- If that isn’t achievable, enable the spill queue so you still record the important crop evidence even when semantics are deferred.

