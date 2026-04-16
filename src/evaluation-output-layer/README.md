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

### Demo

From repo root:

```bash
python src/evaluation-output-layer/demo_evaluation_output.py
```

It prints one JSON line and writes `evaluation_demo.sqlite` next to the demo script.

### Related tooling (not part of the Layer 10 contract)

- **End-to-end video pipeline profiler** (config-driven, prints ROI/YOLO/VLM summaries): `pipeline/benchmark.py`
- **ROI-only studies** (isolated ROI+YOLO timing, ROI on/off matrix): `src/roi-layer/roi_study.py`, `src/roi-layer/roi_benchmark_matrix.py`
