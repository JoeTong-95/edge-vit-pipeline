# Evaluation Output Layer — Changes

## 2026-04-15 (layout)

- Moved the **video pipeline profiler** from `src/evaluation-output-layer/benchmark.py` to `pipeline/benchmark.py` so this folder stays aligned with the Layer 10 contract (`evaluation_output_layer.py` + demo only).
- Moved **ROI-focused benchmark helpers** to the ROI layer:
  - `src/roi-layer/roi_study.py`
  - `src/roi-layer/roi_benchmark_matrix.py` (loads `pipeline/benchmark.py` dynamically)

## 2026-04-15

- Fixed `emit_evaluation_output(...)` to support `output_destination="stdout"` in addition to the SQLite sink.
  - **Problem**: the end-to-end burner smoke test prints evaluation output to stdout and was failing with `Unsupported output_destination: 'stdout'`.
  - **Fix**: allow `"stdout"` and print a deterministic one-line JSON record when selected.

- Renamed the device profiling entrypoint to `benchmark.py` and expanded its output for performance interpretation.
  - Added layer-by-layer summary (Input/ROI/YOLO/VLM) with interpreted throughput/capacity.
  - Added ROI benchmarking helpers (later moved under `src/roi-layer/`; profiler lives at `pipeline/benchmark.py` — see section above).
  - Improved YOLO timing clarity by splitting inference vs post-processing timing, and by reporting pre/post ROI frame counts.

- Updated `pipeline/benchmark.py` VLM timing to support optional micro-batching via `BENCH_VLM_BATCH_SIZE` (defaults to 1, preserving the original behavior).

