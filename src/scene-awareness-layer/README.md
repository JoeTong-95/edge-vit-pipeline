## Scene Awareness Layer (Layer 11)

This folder implements the **Scene Awareness Layer** contract from `pipeline/pipeline_layers_and_interactions.md`.

It is currently a **stub** (no committed model). The goal is to provide a stable interface and a simple, dependency-optional baseline that produces deterministic scene tags from the full frame.

### Files

- `scene_awareness_layer.py`: layer implementation (public functions live here)
- `demo_scene_awareness.py`: small runnable demo that builds a fake `input_layer_package`

### Public API (contract)

The layer exposes:

- `initialize_scene_awareness_layer(config_scene_awareness_enabled: bool, config_device: str="auto") -> dict`
- `run_scene_awareness_inference(scene_awareness_runtime_state: dict, input_layer_package) -> dict | None`
- `build_scene_awareness_layer_package(input_layer_package, raw_result: dict) -> dict`

### Package shape produced

`run_scene_awareness_inference(...)` returns **either**:

- `None` when disabled or when no image is available
- a `scene_awareness_layer_package` dict with **required fields**:
  - `scene_awareness_layer_frame_id`
  - `scene_awareness_layer_timestamp`
  - `scene_awareness_layer_label`
  - `scene_awareness_layer_attributes`
  - `scene_awareness_layer_confidence`

Frame fields are copied from `input_layer_package`:

- `input_layer_frame_id` → `scene_awareness_layer_frame_id`
- `input_layer_timestamp` → `scene_awareness_layer_timestamp`
- `input_layer_image` is used only for inference and is not forwarded by this layer.

### Stub heuristics used

If `numpy` and/or `opencv-python` are available, they are used automatically. If not, a pure-python fallback runs on a downsampled grid.

The stub computes:

- **Brightness**: mean grayscale intensity, mapped to \([0,1]\)
- **Contrast**: grayscale standard deviation, mapped to \([0,1]\)
- **Edge density**:
  - preferred: OpenCV Canny edges and mean(edge>0)
  - fallback: average absolute gradient magnitude proxy
- **Colorfulness**:
  - preferred: per-channel std proxy (normalized)
  - fallback: average channel-difference proxy

These metrics generate coarse tags:

- `dark_scene` / `normal_brightness` / `bright_scene`
- `low_contrast` / `medium_contrast` / `high_contrast`
- `busy_scene` / `calm_scene` (edge density proxy)
- `colorful_scene` / `muted_scene`

The `scene_awareness_layer_label` is a comma-separated list of tags.

### Replacing the stub with a real model

Replace the internal logic in:

- `_compute_scene_metrics(...)` and `_label_from_metrics(...)`

…with your model’s preprocessing + inference + parsing, while keeping the public functions and the output package fields unchanged.

If you add a real model, store an identifier in the runtime state (e.g. `scene_awareness_runtime_model_id`) and propagate it inside `scene_awareness_layer_attributes` so downstream layers can record which model produced each result.

