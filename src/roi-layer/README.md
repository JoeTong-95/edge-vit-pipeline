# ROI Layer

This layer restricts processing to the relevant scene region before detection.

## Public API

- `initialize_roi_layer`
- `update_roi_state`
- `apply_roi_to_frame`
- `build_roi_layer_package`

## Ownership

- Owns startup ROI discovery and the active ROI bounds.
- Passes through the full frame when ROI is disabled or not yet locked.
- Does not depend on tracking for ROI discovery.
- Deduplicates overlapping startup detections locally, so the ROI lock threshold behaves like a unique-ish vehicle count rather than a raw per-frame detection count.

## Quick Start

From the project root:

```powershell
python src/roi-layer/test_roi_layer.py
python src/roi-layer/visualize_roi.py --show
python src/roi-layer/visualize_roi_vlm.py --show
python src/roi-layer/benchmark_roi_yolo.py
python src/roi-layer/roi_study.py
python src/roi-layer/roi_benchmark_matrix.py
python src/roi-layer/render_roi_yolo_gif.py
python src/roi-layer/save_roi_videos.py
```

Other config-driven visualizers that read `src/configuration-layer/config.yaml` are listed in `pipeline/README.md` (including YOLO, tracking, cropper, and end-to-end VLM).

## ROI + VLM Helper

`visualize_roi_vlm.py` is a cross-layer helper around the ROI contract:

- first it runs full-frame YOLO only long enough to calibrate and lock ROI
- once ROI is locked, it runs YOLO only on the ROI crop
- it then selects one ROI-local detection crop and sends that crop to VLM
- the visualization shows the calibration phase, the locked ROI view, the selected crop, and the latest VLM result

## ROI Performance Benchmark

`benchmark_roi_yolo.py` is a headless direct-vs-ROI-cropped comparison helper for YOLO:

- runs `data/sample1.mp4` through `data/sample4.mp4`
- processes up to 30 seconds per trial by default
- runs each sample twice:
  direct full-frame video input and ROI-cropped input
- for the ROI-cropped path, it first does a separate calibration pass to lock ROI,
  then reruns the video with that fixed crop as the YOLO input
- stores per-run and per-frame metrics in SQLite
- generates a dark summary chart comparing end-to-end FPS and inference FPS

Default run:

```powershell
python src/roi-layer/benchmark_roi_yolo.py
```

Useful optional flags:

```powershell
python src/roi-layer/benchmark_roi_yolo.py --max-seconds 20
python src/roi-layer/benchmark_roi_yolo.py --output-dir "E:\OneDrive\desktop\roi-optimization"
python src/roi-layer/benchmark_roi_yolo.py --roi-threshold 5
```

Outputs:

- `roi_eval_metrics_*.sqlite`
- `roi_eval_metrics_*_summary.png`

## ROI Pipeline GIF Renderer

`render_roi_yolo_gif.py` is the separate visual helper for the ROI flow:

- renders full-frame detections during ROI calibration
- shows the locked ROI bounds once calibration completes
- then shows the ROI crop being used as the downstream YOLO input
- writes a GIF to `E:\OneDrive\desktop\roi-optimization`

Default run:

```powershell
python src/roi-layer/render_roi_yolo_gif.py
```

Useful optional flags:

```powershell
python src/roi-layer/render_roi_yolo_gif.py --max-seconds 15
python src/roi-layer/render_roi_yolo_gif.py --video data/sample2.mp4
```

## Batch ROI Video Export

`save_roi_videos.py` batch-saves the `visualize_roi.py` style ROI videos for
`sample1.mp4` through `sample4.mp4`.

Default run:

```powershell
python src/roi-layer/save_roi_videos.py
```

Outputs:

- `sample1_roi_visualized.mp4` through `sample4_roi_visualized.mp4`
- saved to `E:\OneDrive\desktop\roi-optimization`
