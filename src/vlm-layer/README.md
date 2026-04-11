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
  - contains the public interface, package schemas, inference helpers, JSON formatting helpers, and saved debug-image renderer
  - running `python layer.py` prints built-in sample JSON strings and saves built-in sample debug images

- `smoke_test.py`
  - layer-local smoke test
  - creates a burner `VLMFrameCropperLayerPackage` from `truckimage.png`
  - can run either:
    - built-in sample output mode with `--sample-only`
    - one direct VLM inference on `truckimage.png`
  - prints raw, normalized, ack, and combined JSON output
  - saves a debug image that includes the crop, prompt, and actual VLM output

- `run-smoke-test.bat`
  - one-click Windows launcher for the smoke test

- `run_config_vlm_once.py`
  - the real config-driven runner I used for the actual pipeline test
  - reads `src/configuration-layer/config.yaml`
  - runs the actual path:
    - input -> YOLO -> tracking -> VLM cropper cache/dispatch -> VLM
  - saves real debug images with the actual crop, actual prompt, and actual VLM output
  - can stop after:
    - the first real VLM-dispatched track
    - or a target number of unique dispatched track IDs

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
- `build_vlm_output_json` / `format_vlm_output_json` — combines raw result, normalized result, `vlm_layer_package`, and `vlm_ack_package` into one JSON payload.
- `build_sample_vlm_output_json_strings` — returns terminal-friendly sample JSON strings for quick validation without loading the model.
- `save_vlm_debug_image` — saves a debug image that includes the received crop plus the prompt and VLM output text.
- `save_sample_vlm_output_debug_images` — saves debug images for the built-in sample outputs.

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

This now uses a rigid short prompt that asks the model to do only two things:

- decide whether the crop matches one of the currently active YOLO labels such as `truck` or `bus`
- if yes, return JSON with:
  - `wheel_count`
  - `estimated_weight_kg`
  - `ack_status`
  - `retry_reasons`

Current notes:

- `truck_type` is no longer part of the active prompt contract
- model-side `confidence` is no longer requested in the active prompt contract
- `wheel_count` and `estimated_weight_kg` are now expected as integer values
- retry explanation is carried only through `retry_reasons`

## Smoke Test

Run the layer-level smoke test from this folder with:

```bat
run-smoke-test.bat
```

Or directly:

```powershell
python smoke_test.py
```

To print sample JSON strings only, without loading the model:

```powershell
python smoke_test.py --sample-only
python layer.py
```

Both commands now also save debug images by default to:

```text
E:\OneDrive\desktop\01_2026_Projects\01_2026_Cornell_26_Spring\MAE_4221_IoT\DesignProject\Evaluations\07-sample-json-strs
```

To override the directory for the smoke test:

```powershell
python smoke_test.py --sample-only --output-dir path\to\folder
python smoke_test.py --output-dir path\to\folder
```

Useful options:

```powershell
python smoke_test.py --device cpu
python smoke_test.py --track-id demo-123
python smoke_test.py --query-type vehicle_class_only_v1
python smoke_test.py --disabled
```

The smoke test does not depend on the upstream cropper layer. It builds a burner cropper package locally from `truckimage.png` so you can verify this layer in isolation.

## Which File To Run

Use this quick guide:

- `python src/vlm-layer/layer.py`
  - use when you only want built-in sample JSON strings and built-in sample debug images
  - does **not** run the real pipeline
  - does **not** use `config.yaml`

- `python src/vlm-layer/smoke_test.py --sample-only`
  - use when you want the same built-in sample outputs, but through the smoke-test wrapper
  - does **not** run the real pipeline
  - does **not** use `config.yaml`

- `python src/vlm-layer/smoke_test.py`
  - use when you want one direct VLM inference on `truckimage.png`
  - initializes the real model
  - does **not** run YOLO, tracking, or cropper cache logic
  - does **not** use `config.yaml`

- `python src/vlm-layer/run_config_vlm_once.py`
  - use when you want the **real config-driven pipeline**
  - this is the file I ran for the actual test
  - reads `src/configuration-layer/config.yaml`
  - runs input -> YOLO -> tracking -> cropper -> VLM
  - saves the actual crop, actual prompt, and actual VLM response

- `python src/vlm-layer/visualize_vlm.py --show`
  - use when you want the real pipeline plus a live visual debug window
  - best for stepping through behavior interactively

## Real Pipeline Tests

These commands use `src/configuration-layer/config.yaml`.

### 1. One Real Config-Driven VLM Run

Runs until the first real crop is dispatched to VLM, then saves one debug image.

```powershell
python src/vlm-layer/run_config_vlm_once.py --target-ids 1
```

Useful options:

```powershell
python src/vlm-layer/run_config_vlm_once.py --target-ids 1 --device-override cpu
python src/vlm-layer/run_config_vlm_once.py --target-ids 1 --vlm-device-override cpu
python src/vlm-layer/run_config_vlm_once.py --target-ids 1 --max-frames 1000
python src/vlm-layer/run_config_vlm_once.py --target-ids 1 --output-dir path\to\folder
```

### 2. Collect Multiple Real VLM IDs

Runs the real pipeline and saves one debug image per unique VLM-dispatched track ID.

```powershell
python src/vlm-layer/run_config_vlm_once.py --target-ids 15
```

Useful options:

```powershell
python src/vlm-layer/run_config_vlm_once.py --target-ids 5
python src/vlm-layer/run_config_vlm_once.py --target-ids 20 --max-frames 10000
python src/vlm-layer/run_config_vlm_once.py --target-ids 15 --output-dir path\to\folder
```

Notes:

- if the input video does not contain enough eligible tracked vehicles, the run may finish with fewer than the requested number of unique IDs
- `dead_best_available` dispatches are still real cropper/VLM events and are saved too
- when `config_vlm_crop_feedback_enabled` is `false`, the query type becomes `vehicle_semantics_single_shot_v1`
- the current saved debug image now shows only:
  - `track id`
  - `model id`
  - the actual prompt sent to VLM
  - the actual VLM output

### 3. Live Visual Debugging

```powershell
python src/vlm-layer/visualize_vlm.py --show
python src/vlm-layer/visualize_vlm.py --output data/vlm_visualization.mp4
```

Use this when you want to inspect:

- which crop was selected
- when dispatch happened
- what prompt was sent
- what raw text came back
- what ack decision was made

## End-to-end visualizer

From the repo root (requires the same runtime as smoke inference: PyTorch, `transformers`, local model path in `config.yaml`):

```powershell
python src/vlm-layer/visualize_vlm.py --show
python src/vlm-layer/visualize_vlm_roi.py --show
python src/vlm-layer/visualize_vlm_realtime.py --show
python src/vlm-layer/visualize_vlm_roi_realtime.py --show
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

In the current truck-focused flow:

- if VLM decides the crop is **not one of the currently flagged detector
  labels**, it returns an accepted ack with reason `no`; downstream state marks
  that track `no`
- if VLM decides the crop **does match an active target label**, it returns the
  rigid JSON payload and the persistent state marks that track `done`


## Visual Loop Debugging

`visualize_vlm.py` now focuses on the actual decision loop for one track at a time:

- current selected crop
- last image sent to VLM
- VLM judgement and ack
- whether metadata was accepted or the cropper is refilling a new round
- whether a lost object forces VLM to keep using the previous sent image

`visualize_vlm_realtime.py` keeps the input feed moving at source FPS while VLM inference runs on a background worker thread. Use it to see whether the queue/backlog stays under control in a more camera-like setting.

`visualize_vlm_roi.py` adds ROI in front of the same downstream sequence:

- full-frame YOLO is used first only to calibrate and lock ROI
- after ROI lock, YOLO, tracking, cropper selection, and VLM all run on the ROI crop only
- the visualization keeps the cropper/VLM loop view, but now for ROI-local tracks and crops

`visualize_vlm_roi_realtime.py` does the same ROI-integrated path, but uses the
background-worker pattern from `visualize_vlm_realtime.py` so the feed keeps
moving after ROI lock while VLM inference runs asynchronously.


## Retry Reasons

The VLM layer now decides whether the image is good enough by returning structured JSON in `vehicle_semantics_v1`.

Expected behavior:

- if usable: return semantic JSON and `ack_status=accepted`
- if not usable: return `ack_status=retry_requested` plus one or more retry reasons from `occluded` or `bad_angle`

The visualizer shows those retry reasons directly, and only writes metadata after an accepted response.

`vlm_image_quality_notes` is no longer part of the active VLM contract in this branch.


When `config_vlm_crop_feedback_enabled` is `false`, `visualize_vlm.py` runs VLM in single-shot mode: one full crop round, one dispatch, one JSON classification, then the track is marked progressed and skipped by cropper/VLM on future frames.

The visualizer also uses a single-shot query when the cropper dispatches a
`dead_best_available` package after a track stays lost for
`config_vlm_dead_after_lost_frames`. That gives one last best-effort answer for
an incomplete cache round instead of dropping the track entirely.
