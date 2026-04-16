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
python src/roi-layer/test/test_roi_layer.py
python src/roi-layer/util/visualize_roi.py --show
python src/roi-layer/util/visualize_roi_vlm.py --show
python src/roi-layer/test/benchmark_roi_yolo.py
python src/roi-layer/test/roi_study.py
python src/roi-layer/test/roi_benchmark_matrix.py
python src/roi-layer/util/render_roi_yolo_gif.py
python src/roi-layer/util/save_roi_videos.py
```

Other config-driven visualizers that read `src/configuration-layer/config.yaml` are listed in `pipeline/README.md` (including YOLO, tracking, cropper, and end-to-end VLM).

## ROI + VLM Helper

`util/visualize_roi_vlm.py` is a cross-layer helper around the ROI contract:

- first it runs full-frame YOLO only long enough to calibrate and lock ROI
- once ROI is locked, it runs YOLO only on the ROI crop
- it then selects one ROI-local detection crop and sends that crop to VLM
- the visualization shows the calibration phase, the locked ROI view, the selected crop, and the latest VLM result

## ROI Performance Benchmark

`test/benchmark_roi_yolo.py` is a headless direct-vs-ROI-cropped comparison helper for YOLO:

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
python src/roi-layer/test/benchmark_roi_yolo.py
```

Useful optional flags:

```powershell
python src/roi-layer/test/benchmark_roi_yolo.py --max-seconds 20
python src/roi-layer/test/benchmark_roi_yolo.py --output-dir "E:\OneDrive\desktop\roi-optimization"
python src/roi-layer/test/benchmark_roi_yolo.py --roi-threshold 5
```

Outputs:

- `roi_eval_metrics_*.sqlite`
- `roi_eval_metrics_*_summary.png`

## ROI Pipeline GIF Renderer

`util/render_roi_yolo_gif.py` is the separate visual helper for the ROI flow:

- renders full-frame detections during ROI calibration
- shows the locked ROI bounds once calibration completes
- then shows the ROI crop being used as the downstream YOLO input
- writes a GIF to `E:\OneDrive\desktop\roi-optimization`

Default run:

```powershell
python src/roi-layer/util/render_roi_yolo_gif.py
```

Useful optional flags:

```powershell
python src/roi-layer/util/render_roi_yolo_gif.py --max-seconds 15
python src/roi-layer/util/render_roi_yolo_gif.py --video data/sample2.mp4
```

## Batch ROI Video Export

`util/save_roi_videos.py` batch-saves the `util/visualize_roi.py` style ROI videos for
`sample1.mp4` through `sample4.mp4`.

Default run:

```powershell
python src/roi-layer/util/save_roi_videos.py
```

Outputs:

- `sample1_roi_visualized.mp4` through `sample4_roi_visualized.mp4`
- saved to `E:\OneDrive\desktop\roi-optimization`

## Changes

Layer changes in this branch

## 2026-04-15

- Moved utility scripts into `src/roi-layer/util/` and benchmark/test scripts into `src/roi-layer/test/` so the layer root keeps the pipeline-contract implementation prominent.
- Added `test/roi_study.py` and `test/roi_benchmark_matrix.py` (moved from `src/evaluation-output-layer/`) for ROI+YOLO isolation and ROI on/off matrix runs against `pipeline/benchmark.py`.
- Added `ROI_BENCHMARK_NOTES.md` documenting why ROI sometimes appears to provide little YOLO speedup in end-to-end runs, and how to benchmark ROI correctly (post-lock sampling + infer vs post-processing timing).

- Added `roi_layer.py` implementing the pipeline contract public API:
  `initialize_roi_layer`, `update_roi_state`, `apply_roi_to_frame`, and
  `build_roi_layer_package`.
- Implemented the documented ROI node helpers for candidate-box collection,
  ROI-bound computation, bound locking, cropping, and passthrough behavior.
- Added integration compatibility so ROI accepts both dict-style input fixtures
  and the real `InputLayerPackage` dataclass emitted by the input layer on the
  confirmed main path.
- Added `test/test_roi_layer.py` covering disabled passthrough, startup discovery,
  ROI lock, cropped output, and dataclass-package compatibility.
- Added `util/visualize_roi.py` to render ROI discovery, ROI lock state, and a live
  side-by-side view of the full frame and active ROI crop.
- Updated `util/visualize_roi.py` so it only writes a video when `--output` is provided, instead of auto-saving into `data/` on every run.

## 2026-04-09

- Updated `README.md` with `util/visualize_roi.py` quick start and a pointer to `pipeline/README.md` for the full helper list.
- Changed ROI discovery counting so overlapping repeat detections are deduplicated inside the ROI layer before thresholding; this makes `config_roi_vehicle_count_threshold` behave closer to a unique startup-vehicle count.

## 2026-04-10

- Added `util/visualize_roi_vlm.py`, a cross-layer helper that shows ROI calibration
  first and then, after ROI lock, runs YOLO and VLM only inside the cropped ROI
  region.
- Updated `README.md` to document the new ROI -> VLM helper and its phase
  change from calibration to ROI-only downstream inference.
- Added `test/benchmark_roi_yolo.py`, a batch ROI-vs-full-frame benchmark that runs
  sample videos 1-4, logs per-run and per-frame performance into SQLite, and
  writes a styled ROI comparison chart.
- Split visual GIF rendering out of the benchmark path into
  `util/render_roi_yolo_gif.py`, so benchmarking stays headless and does not spend
  time on annotation or GIF export inside the measured loop.
- Refined `test/benchmark_roi_yolo.py` so the comparison now matches the explicit
  evaluation request: direct full-video YOLO input versus a separately
  calibrated fixed ROI crop used as the YOLO input.
- Added a `--roi-threshold` override to `test/benchmark_roi_yolo.py` so ROI
  calibration sweeps can be run without editing `config.yaml`.
- Updated `README.md` with the new headless ROI benchmark behavior and the
  separate ROI pipeline GIF renderer.
- Added `util/save_roi_videos.py`, a batch helper that saves the `util/visualize_roi.py`
  style ROI calibration/crop videos for sample1-4 into the desktop ROI output
  folder.
