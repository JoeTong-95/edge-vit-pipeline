# Tracking Layer

This layer owns object persistence and lifecycle labeling on top of YOLO detections.

## Main Files

- `tracker.py`: tracking layer public API and ByteTrack-backed state
- `util/visualize_tracking.py`: config-driven live or offline visualization for YOLO plus tracking
- `util/automated_evaluation.py`: sequential evaluation sweep across model, device, and tracking mode
- `util/plot_evaluation_results.py`: plotting utility for the evaluation SQLite output
- `test/test_tracking_layer.py`: layer-local validation script

## Public API

The pipeline contract for this layer is implemented in `tracker.py`:
- `initialize_tracking_layer`
- `update_tracks`
- `assign_tracking_status`
- `build_tracking_layer_package`

The emitted package matches the documented flat `tracking_layer_*` field layout.

## Quick Start

From the repo root:

```powershell
python src/tracking-layer/util/visualize_tracking.py
python src/tracking-layer/util/visualize_tracking.py --show
python src/tracking-layer/util/visualize_tracking.py --save-metrics
```

The visualizer reads defaults from `src/configuration-layer/config.yaml` for:
- input video path
- YOLO model
- confidence threshold
- device mode

## Automated Evaluation Sweep

Run the current benchmark sweep with:

```powershell
python src/tracking-layer/util/automated_evaluation.py --max-seconds 60
```

Current sweep combinations:
- device: `cpu`, `cuda`
- model family: `v8`, `v10`, `v11`
- mode: detection only, tracking enabled

Outputs are saved by default to a configurable directory (see `automated_evaluation.py` defaults; override with CLI flags):
- one MP4 per run, such as `eval_v8_cuda_tracking.mp4`
- one SQLite database with run-level and frame-level metrics

## Plotting Evaluation Results

After the sweep, generate a summary plot with:

```powershell
python src/tracking-layer/util/plot_evaluation_results.py
```

The plot utility reads the newest `tracking_eval_metrics*.sqlite` file from the evaluator output directory by default and saves a styled PNG summary next to it.

## Notes

- The tracking visualizer HUD shows mode, end-to-end FPS, inference-only FPS, and track counts.
- SQLite logging is built into both the visualizer and automated evaluator.
- The evaluator handles older SQLite schemas and falls back to a new DB file if the default one is still locked by an older run.
- The end-to-end VLM demo `src/vlm-layer/util/visualize_vlm.py` also consumes this layer’s package shape via the shared `tracker.py` API (see `pipeline/README.md` for helper scope).

## Changes

## 2026-04-08

- Updated `build_tracking_layer_package` in `tracker.py` to emit the documented `tracking_layer_*` fields directly instead of the non-spec `tracking_layer_tracks` wrapper.
- Updated `test_tracking_layer.py` to read the corrected package shape when printing and validating per-track results.
- Updated `visualize_tracking.py` to render and filter tracks from the corrected tracking package contract.
- Kept the public tracking function names unchanged so the layer still matches the pipeline document entry points.
- Updated `visualize_tracking.py` to load video path, model, confidence, and device defaults from `src/configuration-layer/config.yaml`, while still allowing CLI overrides.
- Added `--show` to `visualize_tracking.py` for an optional live preview window.
- Updated `visualize_tracking.py` to show on-screen performance metrics, including live and average FPS plus smoothed YOLO, tracking, draw, write, and loop timings.
- Added device mode and inference-only FPS readouts to `visualize_tracking.py` so the overlay distinguishes full-loop FPS from pure YOLO throughput.
- Removed the last-row timing breakdown from the on-screen tracking overlay to keep the display focused on mode, FPS, and inference FPS.
- Added `automated_evaluation.py` to run the full CPU/CUDA x YOLO v8/v10/v11 x tracking on/off sweep sequentially, save one annotated video per combination, and log both run-level summaries and frame-level metrics to SQLite.
- Set the evaluator defaults to use `src/configuration-layer/config.yaml` for the input video and confidence threshold, while saving batch outputs by default to `E:\OneDrive\desktop\video` with deterministic filenames such as `eval_v8_cuda_tracking.mp4`.
- Added `--max-seconds` to `automated_evaluation.py` so each configuration can be capped by elapsed runtime, which makes one-minute-per-run sweeps deterministic across CPU and CUDA.
- Updated `automated_evaluation.py` to migrate older `evaluation_runs` SQLite tables in place, so existing metrics databases continue working after adding the `max_seconds_requested` field.
- Updated `automated_evaluation.py` to wait on SQLite locks and automatically fall back to a timestamped metrics database file when the default DB is still held by an earlier run.
- Added `plot_evaluation_results.py` to turn the evaluation SQLite output into a dark black-gray scatter plot with an orange palette, including per-frame points and per-run mean overlays saved next to the database file.
- Updated `plot_evaluation_results.py` to fall back to a timestamped PNG filename when the default summary plot path is locked by OneDrive or another process.
- Updated `plot_evaluation_results.py` to use a true legend-style annotation instead of footer text and increased exported summary plot resolution to 300 DPI.
- Added `README.md` for the tracking layer covering the public API, config-driven visualizer, automated evaluation sweep, SQLite output, and summary plotting workflow.
- Updated `visualize_tracking.py` and `automated_evaluation.py` to route frame ingestion through the real `input_layer`, so both video and camera sources are packetized as `input_layer_package` and resized to `config_frame_resolution` before YOLO and tracking.
- Updated `visualize_tracking.py` so annotated video export is opt-in via `--output` instead of auto-writing into `data/` on every run.

## 2026-04-09

- Updated `README.md` to remove hard-coded absolute output paths in prose, and to reference `visualize_vlm.py` as a consumer of the tracking API.
