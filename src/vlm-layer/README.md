# VLM Layer

This folder implements the first working version of the `vlm_layer` described in [pipeline_layers_and_interactions.md](E:\OneDrive\desktop\01_2026_Projects\01_2026_Cornell_26_Spring\MAE_4221_IoT\DesignProject\edge-vlm-pipeline\pipeline\pipeline_layers_and_interactions.md).

## Purpose

The layer adds semantic enrichment beyond detector classes.

It is responsible for:

- loading the configured VLM model
- receiving an object crop package from the cropper layer
- preparing a semantic query
- running inference on the crop
- normalizing the raw model response
- building the downstream `vlm_layer_package`

This layer does **not** own:

- object identity continuity
- new vs existing object decisions
- persistent vehicle state storage
- final metadata output formatting

Those responsibilities stay with the tracking, vehicle state, and metadata layers as specified in the pipeline docs.

## Files

- `layer.py`
  - the self-contained implementation of the VLM layer
  - contains the public interface, package schemas, and inference helpers

- `smoke_test.py`
  - layer-local smoke test
  - creates a burner `VLMFrameCropperLayerPackage` from `truckimage.png`
  - initializes the layer, runs one inference, and prints raw, normalized, and final package output

- `run-smoke-test.bat`
  - one-click Windows launcher for the smoke test

- `truckimage.png`
  - sample image used for the layer smoke test

- `Qwen3.5-0.8B/`
  - local default model artifacts used by this layer implementation
  - currently used as the default `config_vlm_model` path

- `Archive/`
  - previous standalone experimentation scripts and utilities
  - not treated as the active layer implementation
  - should stay in place until the new layer path is fully validated

## Public Interface

The layer exposes the required public functions from the pipeline contract:

- `initialize_vlm_layer`
- `run_vlm_inference`
- `normalize_vlm_result`
- `build_vlm_layer_package`

It also includes the internal node-level helpers named in the pipeline document:

- `prepare_vlm_prompt`
- `infer_vlm_semantics`
- `parse_vlm_response`

## Package Schemas

The implementation defines explicit Python dataclasses for the key package contracts:

- `VLMConfig`
  - includes:
    - `config_vlm_enabled`
    - `config_vlm_model`
    - `config_device`

- `VLMFrameCropperLayerPackage`
  - includes:
    - `vlm_frame_cropper_layer_track_id`
    - `vlm_frame_cropper_layer_image`
    - `vlm_frame_cropper_layer_bbox`

- `VLMRawResult`
  - internal raw result before normalization

- `VLMLayerPackage`
  - includes:
    - `vlm_layer_track_id`
    - `vlm_layer_query_type`
    - `vlm_layer_label`
    - `vlm_layer_attributes`
    - `vlm_layer_confidence`
    - `vlm_layer_model_id`

## Default Query Type

The default query type is:

- `vehicle_semantics_v1`

This uses a constrained semantic prompt that asks the model to return:

- `truck_type`
- `wheel_count`
- `estimated_weight_kg`

That makes early testing easier to compare and normalize.

## Smoke Test

Run the layer-level smoke test from this folder with:

```bat
run-smoke-test.bat
```

Or directly:

```powershell
python smoke_test.py
```

Useful options:

```powershell
python smoke_test.py --device cpu
python smoke_test.py --track-id demo-123
python smoke_test.py --query-type vehicle_class_only_v1
python smoke_test.py --disabled
```

The smoke test does not depend on the upstream cropper layer. It builds a burner cropper package locally from `truckimage.png` so you can verify this layer in isolation.

## Runtime Requirements

I double-checked the repo requirements in [requirements.txt](E:\OneDrive\desktop\01_2026_Projects\01_2026_Cornell_26_Spring\MAE_4221_IoT\DesignProject\edge-vlm-pipeline\docker\requirements.txt).

The current requirement file already includes the main libraries this layer needs:

- `transformers`
- `accelerate`
- `huggingface_hub`
- `safetensors`
- `numpy`
- `pillow`

However, actual inference also requires:

- `torch`

`torch` is not currently listed in the docker requirements file, so the layer code assumes PyTorch is available in the runtime environment.

## Current Scope

This implementation is intentionally limited to the `vlm_layer` folder, following [codex_ground_rules.md](E:\OneDrive\desktop\01_2026_Projects\01_2026_Cornell_26_Spring\MAE_4221_IoT\DesignProject\edge-vlm-pipeline\pipeline\codex_ground_rules.md).

That means:

- this layer now has a real implementation
- upstream cropper integration is still not implemented outside this folder
- downstream vehicle state and metadata integration are still not implemented outside this folder

Those integrations can be added later once the neighboring layers are ready.
