# YOLO Layer Changes

## 2026-04-10

- Added `TAG_FILTER_BEHAVIOR.md` to document the difference between the bundled YOLO weight taxonomy and the repo's active downstream class filter.
- Documented that `class_map.py` is the switchboard for detector tag policy: adding, removing, or renaming entries changes what the Python pipeline keeps after YOLO inference.
- Updated `README.md` to point to `TAG_FILTER_BEHAVIOR.md` as the reference for current forwarded versus discarded detector tags.
- Expanded the current target classes to include `car` alongside `bus` and `truck`, so SUV-like vehicles, pickups, and vans that COCO-style YOLO often collapses into `car` now reach downstream logic.
- Updated `detector.py` so when ROI is active and locked, YOLO uses the actual ROI crop shape as `imgsz` for inference instead of reverting to the model's default square input size. This makes ROI-cropped inference meaningfully different from full-frame inference in compute cost.
- Rounded ROI-driven `imgsz` up to stride-safe multiples inside `detector.py` before calling Ultralytics, so the ROI path no longer emits repeated runtime warnings about auto-adjusted image sizes.

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
