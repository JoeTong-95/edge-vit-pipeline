# YOLO Layer

This layer owns vehicle detection and produces the documented `yolo_layer_package` for downstream tracking.

## Main Files

- `detector.py`: public detection layer functions
- `class_map.py`: class filtering map used by the detector
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
- `cuda` only works if the active Python environment has CUDA-enabled PyTorch.
- The visualizer writes an annotated MP4 by default and can also show a live preview window.
- `src/vlm-layer/visualize_vlm.py` reuses `detector.py` for the same YOLO path inside an end-to-end VLM demo (`pipeline/README.md`).
