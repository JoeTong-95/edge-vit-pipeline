# YOLO Layer

This layer owns vehicle detection and produces the documented `yolo_layer_package` for downstream tracking.

## Main Files

- `detector.py`: public detection layer functions
- `class_map.py`: class filtering map used by the detector
- `TAG_FILTER_BEHAVIOR.md`: explains exactly which YOLO tags are currently forwarded downstream and how editing `class_map.py` changes pipeline behavior
- `visualize_yolo.py`: config-driven visualization helper for YOLO-only runs
- `test_yolo_layer.py`: layer-local test script
- `models/`: bundled local weights such as `yolov8n.pt`, `yolov10n.pt`, and `yolo11n.pt`

## Public API

The pipeline contract for this layer is implemented in `detector.py`:
- `initialize_yolo_layer`
- `run_yolo_detection`
- `filter_yolo_detections`
- `build_yolo_layer_package`

## Quick Start

From `src/yolo-layer`:

```powershell
python .\visualize_yolo.py
python .\visualize_yolo.py --show
python .\visualize_yolo.py --save-metrics
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
- The bundled COCO-pretrained models know many classes, but this layer currently forwards only the project target classes from `class_map.py`: `car`, `bus`, and `truck`. This is intentional so SUV-like vehicles, pickups, and vans that COCO collapses into `car` still reach downstream logic.
- `TAG_FILTER_BEHAVIOR.md` is the explicit reference for what is detected versus discarded in the current repo. Editing `class_map.py` changes detector policy for the Python pipeline and visualizers accordingly.
- `cuda` only works if the active Python environment has CUDA-enabled PyTorch.
- The visualizer writes an annotated MP4 by default and can also show a live preview window.
- `src/vlm-layer/visualize_vlm.py` reuses `detector.py` for the same YOLO path inside an end-to-end VLM demo (`pipeline/README.md`).
- When YOLO receives an active locked `roi_layer_image`, it now uses the crop's native image shape as the inference size instead of silently falling back to the model's default `imgsz`. This makes ROI-cropped inputs materially different from full-frame inputs in performance terms.
