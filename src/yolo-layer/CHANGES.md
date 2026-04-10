# YOLO Layer Changes

## 2026-04-08

- Updated `initialize_yolo_layer` in `detector.py` to prefer bundled local weights from `src/yolo-layer/models/` before falling back to Ultralytics model lookup.
- Added `_resolve_model_path` in `detector.py` so config values like `yolov8n` or `yolov8n.pt` work offline with the repo's local model files.
- Kept the public YOLO function names and `yolo_layer_package` shape unchanged so the layer remains compliant with the pipeline document.
- Added `visualize_yolo.py` to render YOLO detections on video without tracking, with optional annotated output and a live preview window.
- Updated `visualize_yolo.py` to load video path, model, confidence, and device defaults from `src/configuration-layer/config.yaml`, while still allowing CLI overrides.
- Updated `visualize_yolo.py` to show on-screen performance metrics, including live and average FPS plus smoothed inference, draw, write, and loop timings.
- Added device mode and inference-only FPS readouts to `visualize_yolo.py` so the overlay distinguishes full-loop FPS from pure inference throughput.
- Removed the last-row timing breakdown from the on-screen YOLO overlay to keep the display focused on mode, FPS, and inference FPS.
- Added `README.md` for the YOLO layer covering the public API, bundled local models, config-driven visualization flow, and SQLite metrics option in `visualize_yolo.py`.
- Updated `visualize_yolo.py` to initialize and read frames through the real `input_layer`, so video and camera sources are normalized into `input_layer_package` and resized to `config_frame_resolution` before YOLO inference.
- Updated `visualize_yolo.py` so annotated video export is opt-in via `--output` instead of auto-writing into `data/` on every run.

## 2026-04-09

- Updated `README.md` to note `visualize_vlm.py` as another consumer of this layer.
