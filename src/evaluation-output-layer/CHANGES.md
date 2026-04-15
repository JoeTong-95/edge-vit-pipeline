# Evaluation Output Layer — Changes

## 2026-04-15

- Fixed `emit_evaluation_output(...)` to support `output_destination="stdout"` in addition to the SQLite sink.
  - **Problem**: the end-to-end burner smoke test prints evaluation output to stdout and was failing with `Unsupported output_destination: 'stdout'`.
  - **Fix**: allow `"stdout"` and print a deterministic one-line JSON record when selected.

- Renamed the device profiling entrypoint to `benchmark.py` and expanded its output for performance interpretation.
  - Added layer-by-layer summary (Input/ROI/YOLO/VLM) with interpreted throughput/capacity.
  - Added ROI benchmarking helpers:
    - `roi_study.py` (isolated ROI+YOLO measurement)
    - `roi_benchmark_matrix.py` (short ROI on/off runs across sample videos)
  - Improved YOLO timing clarity by splitting inference vs post-processing timing, and by reporting pre/post ROI frame counts.

