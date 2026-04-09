# Tracking Layer Changes

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
- Added README.md for the tracking layer covering the public API, config-driven visualizer, automated evaluation sweep, SQLite output, and summary plotting workflow.

- Updated isualize_tracking.py and utomated_evaluation.py to route frame ingestion through the real input_layer, so both video and camera sources are packetized as input_layer_package and resized to config_frame_resolution before YOLO and tracking.

