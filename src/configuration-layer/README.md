# How Other Layers Should Use This Layer

This section describes the intended usage pattern for the `configuration_layer`.

## Public Entry Point

Other parts of the pipeline should interact with this layer through [`config_node.py`](/e:/OneDrive/desktop/01_2026_Projects/01_2026_Cornell_26_Spring/MAE_4221_IoT/DesignProject/edge-vlm-pipeline/src/configuration-layer/config_node.py).

The public functions are:

- `load_config`
- `validate_config`
- `get_config_value`

you may edit `config.yaml` to change config per run.

All other files under `src/configuration-layer` are internal support files for loading, normalization, typing, defaults, and validation.

## Intended Flow

The expected startup flow is:

1. call `load_config`
2. call `validate_config`
3. pass the required `config_*` values into downstream layer public functions

Example orchestration flow:

```python
config = load_config(config_path)
validate_config(config)

initialize_input_layer(
    config_input_source=get_config_value(config, "config_input_source"),
    config_input_path=get_config_value(config, "config_input_path"),
    config_frame_resolution=get_config_value(config, "config_frame_resolution"),
)
```

## Interaction Boundary

The `configuration_layer` supplies settings.

The `configuration_layer` should not directly call internal functions inside other layers to change their behavior.
Instead, the pipeline orchestrator should read config from this layer and then call each downstream layer's public function with the relevant `config_*` values.

Example:

- `configuration_layer` provides `config_input_source`
- pipeline orchestration reads `config_input_source`
- `input_layer.initialize_input_layer(...)` uses that value to activate `camera_input_node` or `video_file_node`

This keeps the interaction compliant with the pipeline document's boundary: configuration supplies startup and runtime settings, while downstream layers own their own behavior.

## Current Files

Current files in `src/configuration-layer` and their roles:

- `README.md`: explains the layer contract, usage pattern, and current structure
- `config_node.py`: public entry point that exposes `load_config`, `validate_config`, and `get_config_value`
- `config_loader.py`: reads raw config input from a mapping or config file
- `config_normalizer.py`: merges defaults and normalizes raw values into the standard config shape
- `config_validator.py`: checks required parameters, allowed values, and compatibility rules
- `config_types.py`: defines the normalized `ConfigurationLayerConfig` type
- `config_schema.py`: defines the allowed config keys and key rule constants
- `config_defaults.py`: defines default values for the locked `config_*` variables
- `config.yaml`: main editable config file for the current pipeline settings
- `demo_use_setup.py`: runnable example showing how other code should use this layer
- `test_config_node.py`: unit tests for the configuration layer behavior

## Where To Edit Config

If you want to change the current pipeline configuration values, edit `config.yaml`.

Use that file for normal changes such as:

- `config_input_source`
- `config_input_path`
- `config_device`
- `config_vlm_enabled`
- `config_frame_resolution`

Use `config_defaults.py` only if you want to change fallback defaults when a key is missing.
Use `config_schema.py` or `config_validator.py` only if you want to change allowed keys or validation rules.

# Configuration Layer Requirements

This README reflects the locked-down requirements for the configuration layer from `pipeline/pipeline_layers_and_interactions.md`.

## Layer Identity

- Layer name: `configuration_layer`
- Node name: `config_node`

## Purpose

The configuration layer is responsible for:

- selecting runtime mode and enabled layers
- defining the minimum set of behavior-changing parameters

## Public Functions

The configuration layer should expose these public functions exactly:

- `load_config`: read the config source and return normalized config values
- `validate_config`: check required parameters, allowed values, and layer compatibility
- `get_config_value`: return one named config value for downstream layer use

## Locked Configuration Variables

The configuration layer owns and supplies these named config values:

- `config_device`: compute target, such as `cpu` or `cuda`
- `config_input_source`: frame source type, such as `camera` or `video`
- `config_input_path`: file path used when `config_input_source` is `video`
- `config_frame_resolution`: target frame resolution used by the pipeline
- `config_roi_enabled`: enable or disable ROI-based cropping
- `config_roi_vehicle_count_threshold`: number of vehicle detections required before ROI is locked
- `config_yolo_model`: YOLO model variant used for vehicle detection
- `config_yolo_confidence_threshold`: minimum detection confidence accepted from YOLO
- `config_vlm_enabled`: enable or disable the object-level VLM enrichment path
- `config_vlm_model`: VLM model used for semantic inference
- `config_scene_awareness_enabled`: enable or disable the optional full-frame scene-awareness path
- `config_metadata_output_enabled`: enable or disable structured metadata output
- `config_evaluation_output_enabled`: enable or disable evaluation and telemetry output

## Primary Interaction

Primary interaction from the source document:

- supply startup and runtime settings to all active layers

## Required Downstream Usage

The pipeline document explicitly ties these config variables to downstream layers and behaviors:

- `input_layer` uses `config_input_source`, `config_input_path`, and `config_frame_resolution`
- `roi_layer` uses `config_roi_enabled` and `config_roi_vehicle_count_threshold`
- `yolo_layer` uses `config_yolo_model`, `config_yolo_confidence_threshold`, and `config_device`
- `vlm_frame_cropper_layer` uses `config_vlm_enabled`
- `vlm_layer` uses `config_vlm_enabled`, `config_vlm_model`, and `config_device`
- `metadata_output_layer` uses `config_metadata_output_enabled`
- `evaluation_output_layer` uses `config_evaluation_output_enabled`
- `scene_awareness_layer` uses `config_scene_awareness_enabled` and `config_device`

## Naming Constraints

These naming rules from the pipeline document apply to this layer and should be preserved exactly:

- package and variable names should follow `node_name_human_readable_parameter`
- public function names should be action-oriented and stable
- naming should use lowercase `snake_case` consistently

## Scope Boundaries

This layer is responsible for configuration selection, normalization, validation, and retrieval.

This layer is not described as owning:

- frame ingestion
- ROI logic
- detection
- tracking
- semantic inference
- metadata emission
- evaluation emission

Its role is to provide the settings those layers depend on.
