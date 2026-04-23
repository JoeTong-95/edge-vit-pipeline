# Configuration Layer

This layer owns loading, normalizing, validating, and serving the shared `config_*` settings used by the pipeline.

## Public API

Use [`config_node.py`](config_node.py) as the public entry point.

Public functions:
- `load_config`
- `validate_config`
- `get_config_value`

## Main Editable File

Use [`config.yaml`](config.yaml) for normal run-to-run changes.

Common fields:
- `config_input_source`: `video` or `camera`
- `config_input_path`: video path when source is `video`
- `config_device`: `cpu` or `cuda`
- `config_yolo_model`: bundled model such as `yolov8n.pt`, `yolov10n.pt`, `yolov11n.pt`, or `yolov11v28_jingtao.pt`
- `config_yolo_confidence_threshold`: detection confidence threshold
- `config_frame_resolution`: target frame size metadata
- `config_vlm_enabled`: run VLM inference paths when true (requires PyTorch, `transformers`, and a valid `config_vlm_model` path)
- `config_vlm_backend`: VLM backend family selector; use `smolvlm_256m` or `qwen_0_8b` for current local Hugging Face checkpoints, `auto` to infer from the model path
- `config_vlm_model`: filesystem path to the VLM weights (repo-relative paths resolve from the repository root, e.g. `src/vlm-layer/Qwen3.5-0.8B`)
- `config_vlm_crop_feedback_enabled`: when true, VLM may request a better crop round; when false, VLM runs in single-shot mode and the first dispatched image completes that track
- `config_vlm_crop_cache_size`: number of crops collected in one round before dispatch
- `config_vlm_dead_after_lost_frames`: consecutive `lost` tracking updates required before the cropper marks a track dead and finalizes with the best available partial cache
- `config_vlm_runtime_mode`: `inline|async|spill` controls whether VLM runs in the frame loop, in a background worker, or spills overflow requests to disk for offline processing
- `config_vlm_worker_batch_size`: VLM micro-batch size used by async/spill workers (1 disables batching)
- `config_vlm_worker_batch_wait_ms`: batching wait window (ms) to form a micro-batch (async/spill)
- `config_vlm_worker_max_queue_size`: max queued VLM requests before backpressure/spill triggers (async/spill)
- `config_vlm_worker_spill_queue_path`: JSONL path used by spill mode to persist deferred work
- `config_vlm_spill_max_file_mb`: when greater than zero, rotate the active spill file when it reaches this size (MB) so 24/7 runs do not grow one JSONL forever; use `0` only if you manage disk another way
- `config_vlm_realtime_throttle_enabled`: when true, realtime visualizers pace the main loop to the source FPS
- `config_roi_enabled`, `config_roi_vehicle_count_threshold`, and other keys as listed in `config_schema.py`

## Typical Startup Flow

```python
config = load_config(config_path)
validate_config(config)

video_path = get_config_value(config, "config_input_path")
device = get_config_value(config, "config_device")
model_name = get_config_value(config, "config_yolo_model")
```

## Practical Notes

- Relative video paths in `config_input_path` are resolved against the repo root, so `data/sample3.mp4` works even when you launch scripts from a layer subfolder.
- `config_device` accepts `cpu` or `cuda`. Use `cuda` for GPU-backed YOLO inference when your environment supports CUDA.
- The current default config is set up to run against the bundled sample video and local YOLO weights.

## Where This Config Is Used Right Now

- `src/yolo-layer/util/visualize_yolo.py`: YOLO-only (no tracking); draws detections on the input stream.
- `src/tracking-layer/util/visualize_tracking.py`: YOLO + tracking; draws track IDs / statuses on the input stream.
- `src/roi-layer/util/visualize_roi.py`: ROI discovery + lock visualization; shows the active cropped region.
- `src/vlm-frame-cropper-layer/util/visualize_vlm_frame_cropper.py`: Crop-cache visualization; shows per-track candidate crops and when a track becomes dispatchable/dead.
- `src/vlm-layer/util/visualize_vlm.py`: End-to-end cropper → VLM → ack loop for a single focus track (blocking / inline VLM).
- `src/vlm-layer/util/visualize_vlm_realtime.py`: End-to-end VLM loop with a background VLM worker (async); can also run in spill mode to write overflow to `config_vlm_worker_spill_queue_path`.
- `src/vlm-layer/util/visualize_vlm_roi.py`: ROI calibration first, then run YOLO+tracking+cropper+VLM only inside the locked ROI (blocking / inline VLM).
- `src/vlm-layer/util/visualize_vlm_roi_realtime.py`: ROI calibration first, then ROI-cropped realtime async VLM loop (background worker).
- `src/roi-layer/util/visualize_roi_vlm.py`: ROI calibration first, then run YOLO + VLM inside the locked ROI (no full tracking UI; simpler ROI+VLM demo).
- `src/roi-layer/util/visualize_roi_vlm_upson.py`: clip-specific Upson helper that uses the richer ROI -> tracking -> cropper -> VLM UI, starts at `0:48`, ends at `1:48`, and paces playback to the source FPS by default.
- `src/tracking-layer/util/automated_evaluation.py`: headless evaluation run (writes metrics/artifacts; no visualization UI).

These scripts read from `config.yaml` by default and let CLI arguments override specific values.

## Changes

## 2026-04-15

- Added VLM runtime knobs to the shared config contract: `config_vlm_runtime_mode`, `config_vlm_worker_max_queue_size`, `config_vlm_worker_batch_size`, `config_vlm_worker_batch_wait_ms`, `config_vlm_worker_spill_queue_path`, `config_vlm_spill_max_file_mb`, and `config_vlm_realtime_throttle_enabled`.
- Updated defaults / schema / types / normalizer / validator so these knobs are available everywhere that reads `config.yaml`.
- Updated `config.yaml` and `README.md` to document and expose the new knobs for benchmark + realtime VLM tools.
- `config_vlm_spill_max_file_mb` caps the active spill JSONL size by rotating to a sibling `*.jsonl.rotated.<ms>` file (delete or archive rotated files separately for 24/7 deployments).

## 2026-04-16

- Updated the documented bundled YOLO choices to include `yolov11v28_jingtao.pt`.
- Documented `src/roi-layer/util/visualize_roi_vlm_upson.py` as the clip-specific Upson ROI/VLM helper.
- `benchmark.py` now resolves `src/vlm-layer/util/` as well, so `config_vlm_runtime_mode=async` can initialize `AsyncVLMWorker` instead of falling back to inline because of an import miss.

## 2026-04-10

- Added `config_vlm_dead_after_lost_frames` to the shared configuration contract so the cropper can declare a track dead after a configurable number of consecutive `lost` tracking updates.
- Updated `config_defaults.py`, `config_schema.py`, `config_types.py`, and `config_normalizer.py` so the new dead-threshold setting is normalized and carried like the rest of the pipeline config.
- Updated `config_validator.py` so `config_vlm_dead_after_lost_frames` must be greater than 0.
- Updated `config.yaml` to expose the dead-threshold control for the cropper and VLM visualizers.
- Updated `README.md` to document the new dead-threshold setting and how it affects partial-cache finalization.

## 2026-04-08

- Updated `config.yaml` to point to the existing `data/sample3.mp4` file instead of the missing `data/sample1.mp4`.
- Updated the default `config_input_path` in `config_defaults.py` to `data/sample3.mp4` so a default video-mode configuration is runnable from the repo.
- Updated the default `config_yolo_model` in `config_defaults.py` and `config.yaml` to `yolov8n.pt` to match the bundled local YOLO weights.
- Tightened `validate_config` via `config_validator.py` so video-mode configs fail fast when `config_input_path` does not exist.
- Updated `config_validator.py` to resolve relative `config_input_path` values against the repo root as well, so scripts launched from layer subfolders still honor the shared config file.
- Rewrote README.md to document the current config-driven workflow, including config.yaml, accepted config_device values, bundled YOLO model names, and the scripts that now consume this layer by default.

## 2026-04-09

- Added `config_vlm_crop_cache_size` to the shared configuration contract so the cropper-side local cache size can be controlled from `config.yaml`.
- Updated `config_defaults.py`, `config_schema.py`, `config_types.py`, and `config_normalizer.py` so the new cache-size setting is normalized like the rest of the pipeline config.
- Updated `config_validator.py` so `config_vlm_crop_cache_size` must be greater than 0.
- Updated `config.yaml` to expose the new cache-size control for the cropper and cropper visualizer.
- Updated the shared visualizer defaults so ROI, YOLO, tracking, and cropper helpers no longer auto-save videos into `data/` unless `--output` is passed explicitly.
- Documented that `config_vlm_crop_cache_size` now directly controls the one-shot VLM candidate collection window before dispatch.

## 2026-04-09 (documentation and defaults)

- Updated `config.yaml` so `config_vlm_model` points at the bundled `src/vlm-layer/Qwen3.5-0.8B` directory and `config_vlm_enabled` reflects the VLM path used by `visualize_vlm.py` (adjust per machine as needed).
- Refreshed `README.md` with VLM-related keys, relative links, and consumers including `visualize_vlm.py`.
- Added `config_vlm_crop_feedback_enabled` so the optional VLM path can run either as a true feedback loop (`true`) or as single-shot classify-and-complete (`false`).
