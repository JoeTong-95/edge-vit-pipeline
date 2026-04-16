## Metadata Output Layer (Layer 9)

This layer formats and emits structured metadata as the main output product of the pipeline.

### Public API

Implemented in `metadata_output_layer.py`:

- `build_metadata_output_layer_package(vehicle_state_layer_package, vlm_layer_package=None, scene_awareness_layer_package=None) -> dict`
- `serialize_metadata_output(metadata_output_layer_package, output_format="json") -> str | bytes`
- `emit_metadata_output(serialized_payload, output_destination="stdout", output_path=None) -> None`

### Output package (JSON) shape

The emitted JSON is a deterministic, schema-stable dict with these top-level keys (per pipeline contract):

- `metadata_output_layer_timestamps`: list of ISO-8601 UTC timestamps (one per object record)
- `metadata_output_layer_object_ids`: list of object ids (track ids), sorted deterministically
- `metadata_output_layer_classes`: list of stored/derived classes aligned with `object_ids`
- `metadata_output_layer_semantic_tags`: list of tag-lists aligned with `object_ids`
- `metadata_output_layer_scene_tags`: optional list of scene-level tags (empty if absent)
- `metadata_output_layer_counts`: aggregate counts (by class, semantic tag, scene tag)
- `metadata_output_layer_summaries`: generated_at + per-object summary + high-level summary

### Usage

Run the layer smoke test from the repo root:

```bash
python src/metadata-output-layer/test/test_metadata_output_layer.py
```

To embed in the pipeline:

```python
from metadata_output_layer import (
    build_metadata_output_layer_package,
    serialize_metadata_output,
    emit_metadata_output,
)

pkg = build_metadata_output_layer_package(vehicle_state_layer_package, vlm_layer_package, scene_awareness_layer_package)
payload = serialize_metadata_output(pkg, output_format="json")
emit_metadata_output(payload, output_destination="stdout")
```

### Notes

- Standard library only (no `pydantic`, `orjson`, etc.).
- If `vlm_layer_package` or `scene_awareness_layer_package` is missing/empty, the output degrades gracefully.
- Ordering is deterministic (objects sorted by numeric track id when possible; otherwise lexicographic).

