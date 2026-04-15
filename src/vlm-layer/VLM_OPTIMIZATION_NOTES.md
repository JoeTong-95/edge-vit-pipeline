## VLM performance optimization notes (branch: `vlm-optimization`)

This repo’s object-level VLM path is:

`tracking → vlm_frame_cropper(cache/select) → dispatch → VLM(generate) → ack → vehicle_state`

### What was limiting performance

- **No batching**: the core VLM code was effectively batch-size-1 (`processor(text=[...], images=[...])`), so GPU utilization is often poor and per-call overhead is paid for every crop.
- **Backpressure only per-track**: the cropper ensures at most one in-flight request **per track**, but there was no global control; when multiple tracks become dispatchable at once, VLM latency can spike and the worker queue can saturate.
- **When overloaded, you need a policy**: either keep up in realtime (by increasing throughput / reducing call rate) or intentionally defer VLM work (“cache-and-run”) so you don’t miss transient events.

### Can the current stack keep up in realtime?

It depends on two numbers:

- **dispatch rate** (\(\lambda\)): crops/sec sent to VLM
- **service time** (\(T\)): seconds per VLM query

To keep up in steady state, you need \(\lambda \cdot T < 1\) (or, equivalently, \(T < 1/\lambda\)).

Practical guidance in this repo:

- Use `src/evaluation-output-layer/benchmark.py` to measure end-to-end dispatch rate and per-stage latency; it already prints a “target VLM query time to keep up”.
- If dispatch rate is high (many vehicles / frequent retries), the best knobs to reduce \(\lambda\) are:
  - `config_roi_enabled` (reduces upstream load)
  - `config_vlm_crop_cache_size` (dispatch less often; also increases “time to first VLM”)
  - `config_vlm_crop_feedback_enabled=false` (single-shot; fewer retries)
  - reduce crop size/padding (`VLM_CROP_CONTEXT_PADDING_RATIO` in `vlm_frame_cropper_layer.py`)

### Changes implemented on this branch

#### 1) Batched VLM inference API

Added true batch inference utilities in `src/vlm-layer/layer.py`:

- `infer_vlm_semantics_batch(...)`
- `run_vlm_inference_batch(...)`

These run a single `model.generate(...)` over a batch of crops and decode each item separately.

#### 2) Micro-batching in the realtime async worker

Updated `src/vlm-layer/visualize_vlm_realtime.py` `AsyncVLMWorker`:

- It now collects up to `--vlm-batch-size` tasks (within `--vlm-batch-wait-ms`) and runs **one** batched VLM call.
- This typically increases throughput on GPU by amortizing overhead.

New CLI flags:

- `--vlm-batch-size` (default `1`)
- `--vlm-batch-wait-ms` (default `20`)

These automatically apply to `visualize_vlm_roi_realtime.py` too (it imports the same worker).

#### 3) “Cache-and-run” overflow spill queue (optional)

When realtime can’t keep up, you can switch to “cache-and-run” behavior without freezing tracks:

- If the worker queue is full and `--vlm-spill-queue` is set, the task is **persisted** to a JSONL file with an embedded PNG crop.
- The worker also emits an immediate **accepted** ack with reason `deferred_spill_to_queue`, so the cropper’s per-track in-flight flag is cleared and the pipeline continues.

Files:

- `src/vlm-layer/vlm_deferred_queue.py` (JSONL format + PNG base64 encode/decode)
- `src/vlm-layer/run_deferred_vlm_queue.py` (offline processor)

Realtime usage example:

```bash
python src/vlm-layer/visualize_vlm_realtime.py --input-source video --video data/sample4.mp4 ^
  --vlm-batch-size 6 --vlm-batch-wait-ms 25 ^
  --max-queue-size 64 ^
  --vlm-spill-queue data/deferred_vlm_queue.jsonl
```

Offline processing example:

```bash
python src/vlm-layer/run_deferred_vlm_queue.py ^
  --queue data/deferred_vlm_queue.jsonl ^
  --out data/deferred_vlm_results.jsonl ^
  --model src/vlm-layer/Qwen3.5-0.8B ^
  --device cuda ^
  --batch-size 8
```

### Recommended operating modes

- **Realtime-first** (try to keep up):
  - Enable ROI (`config_roi_enabled=true`)
  - Use micro-batching (`--vlm-batch-size` 4–8, `--vlm-batch-wait-ms` 10–30)
  - Consider disabling feedback (`config_vlm_crop_feedback_enabled=false`) if retries dominate.

- **Cache-and-run** (when realtime VLM is impossible on device):
  - Set `--vlm-spill-queue` to persist crops
  - Process later with `run_deferred_vlm_queue.py`
  - This prevents “truck crossed the frame before VLM returned” from losing the evidence: the crop is stored even if semantics are computed later.

