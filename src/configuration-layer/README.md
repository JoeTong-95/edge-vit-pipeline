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

- `src/yolo-layer/visualize_yolo.py`
- `src/tracking-layer/visualize_tracking.py`
- `src/tracking-layer/automated_evaluation.py`
- `src/roi-layer/visualize_roi.py`
- `src/vlm-frame-cropper-layer/visualize_vlm_frame_cropper.py`
- `src/vlm-layer/visualize_vlm.py`

These scripts read from `config.yaml` by default and let CLI arguments override specific values.
