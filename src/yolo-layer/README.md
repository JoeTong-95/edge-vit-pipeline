# YOLO Layer

This layer owns vehicle detection and produces the documented `yolo_layer_package` for downstream tracking.

## Main Files

- `detector.py`: public detection layer functions
- `class_map.py`: class filtering map used by the detector
- `TAG_FILTER_BEHAVIOR.md`: explains exactly which YOLO tags are currently forwarded downstream and how editing `class_map.py` changes pipeline behavior
- `util/visualize_yolo.py`: config-driven visualization helper for YOLO-only runs
- `test/test_yolo_layer.py`: layer-local test script
- `models/`: bundled local weights such as `yolov8n.pt`, `yolov10n.pt`, `yolov11n.pt`, and `yolov11v28_jingtao.pt`

## Public API

The pipeline contract for this layer is implemented in `detector.py`:
- `initialize_yolo_layer`
- `run_yolo_detection`
- `filter_yolo_detections`
- `build_yolo_layer_package`

## Quick Start

From the repo root:

```powershell
python src/yolo-layer/util/visualize_yolo.py
python src/yolo-layer/util/visualize_yolo.py --show
python src/yolo-layer/util/visualize_yolo.py --save-metrics
```

By default, the script reads these values from `src/configuration-layer/config.yaml`:
- input video path
- YOLO model
- confidence threshold
- device mode

## What The Visualizer Shows

The on-screen HUD includes:
- frame id
- mode: `cpu` or `cuda`
- end-to-end FPS
- inference-only FPS
- detection count

If `--save-metrics` is enabled, the script stores run metadata and frame-level metrics in SQLite.

## Notes

- Bundled local weights are preferred before Ultralytics falls back to external lookup.
- `class_map.py` is model-specific policy, not a universal COCO truth table. The current active map is tuned for `yolov11v28_jingtao.pt`, which uses a custom vehicle taxonomy (`pickup_truck`, `bus`, `van`, `tow_truck`, `semi_truck`, `box_truck`, `dump_truck`) rather than COCO IDs.
- `TAG_FILTER_BEHAVIOR.md` is the explicit reference for what is detected versus discarded in the current repo. Editing `class_map.py` changes detector policy for the Python pipeline and visualizers accordingly.
- `cuda` only works if the active Python environment has CUDA-enabled PyTorch.
- The visualizer writes an annotated MP4 by default and can also show a live preview window.
- `src/vlm-layer/util/visualize_vlm.py` reuses `detector.py` for the same YOLO path inside an end-to-end VLM demo (`pipeline/README.md`).
- When YOLO receives an active locked `roi_layer_image`, it now uses the crop's native image shape as the inference size instead of silently falling back to the model's default `imgsz`. This makes ROI-cropped inputs materially different from full-frame inputs in performance terms.

## Changes

## 2026-04-16

- Added bundled custom checkpoint `src/yolo-layer/models/yolov11v28_jingtao.pt`.
- Updated `class_map.py` so the active forwarded classes match the Jingtao checkpoint taxonomy instead of the previous COCO-style IDs.
- Kept the previous COCO-style map commented in `class_map.py` for reference so it is still easy to switch back when testing older bundled weights.

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
