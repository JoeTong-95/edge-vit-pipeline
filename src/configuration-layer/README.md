# Configuration Layer

This layer owns loading, normalizing, validating, and serving the shared `config_*` settings used by the pipeline.

## Public API

Use [`config_node.py`](/e:/OneDrive/desktop/01_2026_Projects/01_2026_Cornell_26_Spring/MAE_4221_IoT/DesignProject/edge-vlm-pipeline/src/configuration-layer/config_node.py) as the public entry point.

Public functions:
- `load_config`
- `validate_config`
- `get_config_value`

## Main Editable File

Use [`config.yaml`](/e:/OneDrive/desktop/01_2026_Projects/01_2026_Cornell_26_Spring/MAE_4221_IoT/DesignProject/edge-vlm-pipeline/src/configuration-layer/config.yaml) for normal run-to-run changes.

Common fields:
- `config_input_source`: `video` or `camera`
- `config_input_path`: video path when source is `video`
- `config_device`: `cpu` or `cuda`
- `config_yolo_model`: bundled model such as `yolov8n.pt`, `yolov10n.pt`, or `yolo11n.pt`
- `config_yolo_confidence_threshold`: detection confidence threshold
- `config_frame_resolution`: target frame size metadata

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

These scripts read from `config.yaml` by default and let CLI arguments override specific values.
