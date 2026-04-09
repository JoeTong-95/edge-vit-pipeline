# Configuration Layer Changes

## 2026-04-08

- Updated `config.yaml` to point to the existing `data/sample.mp4` file instead of the missing `data/sample1.mp4`.
- Updated the default `config_input_path` in `config_defaults.py` to `data/sample.mp4` so a default video-mode configuration is runnable from the repo.
- Updated the default `config_yolo_model` in `config_defaults.py` and `config.yaml` to `yolov8n.pt` to match the bundled local YOLO weights.
- Tightened `validate_config` via `config_validator.py` so video-mode configs fail fast when `config_input_path` does not exist.
- Updated `config_validator.py` to resolve relative `config_input_path` values against the repo root as well, so scripts launched from layer subfolders still honor the shared config file.
- Rewrote README.md to document the current config-driven workflow, including config.yaml, accepted config_device values, bundled YOLO model names, and the scripts that now consume this layer by default.

