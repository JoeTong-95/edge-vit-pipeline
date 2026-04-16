## Evaluation Output Layer (Layer 10)

Implements the **Evaluation Output Layer** contract from `pipeline/pipeline_layers_and_interactions.md`.

### Public API

Implemented in `evaluation_output_layer.py`:

- `collect_evaluation_metrics(...) -> dict`
- `build_evaluation_output_layer_package(metrics: dict) -> dict`
- `emit_evaluation_output(evaluation_output_layer_package, output_destination="sqlite", output_path="evaluation.sqlite") -> None`

### Output package fields (contract)

- `evaluation_output_layer_fps`
- `evaluation_output_layer_module_latency`
- `evaluation_output_layer_detection_count`
- `evaluation_output_layer_track_count`
- `evaluation_output_layer_vlm_call_count`
- `evaluation_output_layer_scene_call_count`

This implementation also adds optional debug fields:

- `evaluation_output_layer_frame_id`
- `evaluation_output_layer_timestamp`
- `evaluation_output_layer_collected_at` (ISO-8601 UTC)

### SQLite emission

Default sink is `sqlite3` (stdlib). See `evaluation_output_layer.py` for table names and column layout.

### Related tooling (not part of the Layer 10 contract)

- **End-to-end video pipeline profiler** (config-driven, prints ROI/YOLO/VLM summaries): `pipeline/benchmark.py`
- **ROI-only studies** (isolated ROI+YOLO timing, ROI on/off matrix): `src/roi-layer/roi_study.py`, `src/roi-layer/roi_benchmark_matrix.py`

## Changes

## 2026-04-15 (layout)

- Removed the standalone demo script so this folder stays aligned with the Layer 10 contract (`evaluation_output_layer.py` plus the public package export).
- Moved the **video pipeline profiler** from `src/evaluation-output-layer/benchmark.py` to `pipeline/benchmark.py`.
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
