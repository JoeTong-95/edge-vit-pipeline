# VLM Layer

This folder implements the working `vlm_layer` described in [`../../pipeline/pipeline_layers_and_interactions.md`](../../pipeline/pipeline_layers_and_interactions.md).

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

- `visualize_vlm.py`
  - orchestration helper (not a replacement for the layer contract): reads `src/configuration-layer/config.yaml`, runs input → YOLO → tracking → cropper cache/dispatch → `run_vlm_inference`, then `register_vlm_ack_package` and vehicle-state ack updates
  - three-column output: raw frame + tracks, cropper cache and selected crop, and a scrollable log of dispatch mode/reason, user prompt, chat-template preview (`preview_vlm_applied_prompt`), model response, and ack status
  - options include `--show`, `--output`, `--retry-on-unknown`, `--demo-first-retry`, `--vlm-device`

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
- `build_vlm_ack_package`

Additional helper used by `visualize_vlm.py` and debugging:

- `preview_vlm_applied_prompt` — returns the processor chat-template string for a given user prompt (same shape as inference).

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

- `VLMAckPackage`
  - includes:
    - `vlm_ack_track_id`
    - `vlm_ack_status`
    - `vlm_ack_reason`
    - `vlm_ack_retry_requested`

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

## End-to-end visualizer

From the repo root (requires the same runtime as smoke inference: PyTorch, `transformers`, local model path in `config.yaml`):

```powershell
python src/vlm-layer/visualize_vlm.py --show
python src/vlm-layer/visualize_vlm_realtime.py --show
python src/vlm-layer/visualize_vlm.py --output data/vlm_visualization.mp4
python src/vlm-layer/visualize_vlm_realtime.py --show --max-queue-size 128
```

## Runtime Requirements

The shared dependency list is in [`../../docker/requirements.txt`](../../docker/requirements.txt).

The current requirement file already includes the main libraries this layer needs:

- `transformers`
- `accelerate`
- `huggingface_hub`
- `safetensors`
- `numpy`
- `pillow`

However, actual inference also requires:

- `torch`

The bundled **Qwen3.5** checkpoint (`model_type` **`qwen3_5`**) requires **`transformers` 5.x** (4.x does not register that architecture). Use `pip install -U "transformers>=5.0.0,<6.0.0"` if you see errors mentioning `qwen3_5` or an unrecognized model architecture.

`torch` is not listed in `docker/requirements.txt`; the layer assumes PyTorch is available in the runtime environment.

## Scope and orchestration

Core inference and packages live in this folder per [`../../pipeline/codex_ground_rules.md`](../../pipeline/codex_ground_rules.md). `visualize_vlm.py` is documented as a **cross-layer helper** (same category as other `visualize_*.py` scripts in `pipeline/README.md`): it calls neighboring layer APIs to demonstrate collect → dispatch → infer → ack without changing their contracts.


## Ack Loop Role

The VLM layer now also returns a lightweight acknowledgement package after one dispatched crop is reviewed. That acknowledgement lets the cropper and vehicle-state layers distinguish between:

- `accepted`: the selected crop is usable and the track is finalized
- `retry_requested`: the current crop is not good enough and the cropper may reopen selection if the track is still visible
- `finalize_with_current`: the object has left scope and downstream logic should work with the best crop already available


## Visual Loop Debugging

`visualize_vlm.py` now focuses on the actual decision loop for one track at a time:

- current selected crop
- last image sent to VLM
- VLM judgement and ack
- whether metadata was accepted or the cropper is refilling a new round
- whether a lost object forces VLM to keep using the previous sent image

`visualize_vlm_realtime.py` keeps the input feed moving at source FPS while VLM inference runs on a background worker thread. Use it to see whether the queue/backlog stays under control in a more camera-like setting.


## Retry Reasons

The VLM layer now decides whether the image is good enough by returning structured JSON in `vehicle_semantics_v1`.

Expected behavior:

- if usable: return semantic JSON and `ack_status=accepted`
- if not usable: return `ack_status=retry_requested` plus one or more retry reasons from `occluded` or `bad_angle`

The visualizer shows those retry reasons directly, and only writes metadata after an accepted response.


When `config_vlm_crop_feedback_enabled` is `false`, `visualize_vlm.py` runs VLM in single-shot mode: one full crop round, one dispatch, one JSON classification, then the track is marked progressed and skipped by cropper/VLM on future frames.
