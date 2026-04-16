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
- `config_yolo_model`: bundled model such as `yolov8n.pt`, `yolov10n.pt`, or `yolo11n.pt`
- `config_yolo_confidence_threshold`: detection confidence threshold
- `config_frame_resolution`: target frame size metadata
- `config_vlm_enabled`: run VLM inference paths when true (requires PyTorch, `transformers`, and a valid `config_vlm_model` path)
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

- Relative video paths in `config_input_path` are resolved against the repo root, so `data/sample.mp4` works even when you launch scripts from a layer subfolder.
- `config_device` accepts `cpu` or `cuda`. Use `cuda` for GPU-backed YOLO inference when your environment supports CUDA.
- The current default config is set up to run against the bundled sample video and local YOLO weights.

## Where This Config Is Used Right Now

- `src/yolo-layer/visualize_yolo.py`: YOLO-only (no tracking); draws detections on the input stream.
- `src/tracking-layer/visualize_tracking.py`: YOLO + tracking; draws track IDs / statuses on the input stream.
- `src/roi-layer/visualize_roi.py`: ROI discovery + lock visualization; shows the active cropped region.
- `src/vlm-frame-cropper-layer/visualize_vlm_frame_cropper.py`: Crop-cache visualization; shows per-track candidate crops and when a track becomes dispatchable/dead.
- `src/vlm-layer/visualize_vlm.py`: End-to-end cropper → VLM → ack loop for a single focus track (blocking / inline VLM).
- `src/vlm-layer/visualize_vlm_realtime.py`: End-to-end VLM loop with a background VLM worker (async); can also run in spill mode to write overflow to `config_vlm_worker_spill_queue_path`.
- `src/vlm-layer/visualize_vlm_roi.py`: ROI calibration first, then run YOLO+tracking+cropper+VLM only inside the locked ROI (blocking / inline VLM).
- `src/vlm-layer/visualize_vlm_roi_realtime.py`: ROI calibration first, then ROI-cropped realtime async VLM loop (background worker).
- `src/roi-layer/visualize_roi_vlm.py`: ROI calibration first, then run YOLO + VLM inside the locked ROI (no full tracking UI; simpler ROI+VLM demo).
- `src/tracking-layer/automated_evaluation.py`: headless evaluation run (writes metrics/artifacts; no visualization UI).

These scripts read from `config.yaml` by default and let CLI arguments override specific values.
