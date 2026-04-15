## Evaluation Output Layer (Layer 10)

Implements the **Evaluation Output Layer** contract from `pipeline/pipeline_layers_and_interactions.md`.

### Public API

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

### SQLite emission

Default sink is `sqlite3` (stdlib). Table is created if missing:

- Table: `evaluation_records`

Columns:

- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `created_at_s` REAL NOT NULL
- `frame_id` TEXT
- `fps` REAL
- `detection_count` INTEGER NOT NULL
- `track_count` INTEGER NOT NULL
- `vlm_call_count` INTEGER NOT NULL
- `scene_call_count` INTEGER NOT NULL
- `module_latency_json` TEXT NOT NULL
- `evaluation_package_json` TEXT NOT NULL

### Demo

Run from this folder:

```bash
python demo_evaluation_output.py
```

It prints one JSON line and writes `evaluation_demo.sqlite` in the same directory.

## Evaluation Output Layer (Layer 10)

This layer collects lightweight benchmarking / debug telemetry and emits it to a sink.

Contract source: `pipeline/pipeline_layers_and_interactions.md` (Layer 10).

### Public API

Implemented in `evaluation_output_layer.py`:

- `collect_evaluation_metrics(input_layer_package=None, roi_layer_package=None, yolo_layer_package=None, tracking_layer_package=None, vlm_layer_package=None, scene_awareness_layer_package=None, timings=None) -> dict`
- `build_evaluation_output_layer_package(metrics: dict) -> dict`
- `emit_evaluation_output(evaluation_output_layer_package: dict, output_destination="sqlite", output_path="evaluation.sqlite") -> None`

### Output package fields (per contract)

`build_evaluation_output_layer_package()` returns:

- `evaluation_output_layer_fps`
- `evaluation_output_layer_module_latency`
- `evaluation_output_layer_detection_count`
- `evaluation_output_layer_track_count`
- `evaluation_output_layer_vlm_call_count`
- `evaluation_output_layer_scene_call_count`

This implementation also adds:

- `evaluation_output_layer_collected_at` (ISO-8601 UTC)

### SQLite schema

The `sqlite` emitter writes to table `evaluation_output_v1`:

- `collected_at` (TEXT)
- `fps` (REAL)
- `module_latency_json` (TEXT, JSON)
- `detection_count` (INTEGER)
- `track_count` (INTEGER)
- `vlm_call_count` (INTEGER)
- `scene_call_count` (INTEGER)

### Demo

From repo root (or this folder):

```bash
python src/evaluation-output-layer/demo_evaluation_output.py
```

