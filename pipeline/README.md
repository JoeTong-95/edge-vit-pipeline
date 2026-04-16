This folder defines the locked pipeline contracts and ownership boundaries for the repo.

## Key Files

- `pipeline_layers_and_interactions.md`: the main source of truth for layer names, public functions, package shapes, and interactions.
- `codex_ground_rules.md`: repo-specific working constraints, including the rule to stay inside one layer unless explicitly asked.
- `master-pipeline.png`: visual reference for the pipeline structure.

## Current Practical Flow

The currently runnable path in this branch is:

`configuration_layer -> input_layer -> yolo_layer -> tracking_layer`

When `config_vlm_enabled` is true and the runtime has PyTorch plus `transformers`, helper scripts can also exercise the **optional** path through the cropper and `vlm_layer` (collect → dispatch → infer → ack). The contract still comes from `pipeline_layers_and_interactions.md`, even when local helper scripts are used for visualization or evaluation.

## Current Utility Scripts Added Around The Contract

These scripts are helpers around the layer APIs, not replacements for the layer contract itself:

- `src/yolo-layer/util/visualize_yolo.py`: config-driven YOLO-only visualization with optional live preview, optional video output, and optional SQLite metrics.
- `src/tracking-layer/util/visualize_tracking.py`: config-driven YOLO plus tracking visualization with optional live preview, optional video output, and optional SQLite metrics.
- `src/roi-layer/util/visualize_roi.py`: config-driven ROI discovery visualization with optional live preview and optional video output.
- `src/roi-layer/util/visualize_roi_vlm.py`: calibrates ROI from full-frame detections first, then after ROI lock runs YOLO and VLM only inside the ROI crop and shows the selected ROI-local crop sent to VLM.
- `src/roi-layer/util/visualize_roi_vlm_upson.py`: clip-specific Upson helper; runs the richer ROI -> tracking -> cropper -> VLM path from `0:48` to `1:48`, keeps playback paced to source FPS by default, and keeps the left display anchored to the original full-frame aspect ratio after ROI lock.
- `src/vlm-frame-cropper-layer/util/visualize_vlm_frame_cropper.py`: config-driven crop-cache visualization showing per-track crop history and the current crop selected for VLM input.
- `src/vlm-layer/util/visualize_vlm.py`: end-to-end orchestration visualizer (input → YOLO → tracking → cropper cache/dispatch → VLM inference) with structured prompt preview, raw response, and ack/retry logging; reads `config.yaml` like the other helpers.
- `src/vlm-layer/util/visualize_vlm_roi.py`: calibrates ROI first, then runs the downstream tracking -> cropper cache/selection -> VLM loop only inside the locked ROI crop.
- `src/vlm-layer/util/visualize_vlm_realtime.py`: non-blocking VLM helper that keeps the feed running at source FPS while VLM inference happens on a background worker, making it easier to see backlog and whether inference keeps up with a live-like stream.
- `src/vlm-layer/util/visualize_vlm_roi_realtime.py`: ROI-integrated non-blocking VLM helper; calibrates ROI first, then keeps the ROI-local feed moving while cropper-selected VLM inference runs on a background worker.
- `src/tracking-layer/util/automated_evaluation.py`: sequential benchmark sweep across CPU/CUDA, YOLO v8/v10/v11, and tracking on/off.
- `src/tracking-layer/util/plot_evaluation_results.py`: creates styled summary plots from the evaluation SQLite output.

## Running on Jetson

See `JETSON_OPTIMIZATION.md` for the full session record. Quick-start:

```bash
# Install Jetson-optimized torch stack (run once)
bash docker/setup-native-jetson.sh

# Run benchmark with Jetson config (TRT FP16 engine, VLM on CPU)
BENCH_CONFIG_YAML=src/configuration-layer/config.jetson.yaml python3 benchmark.py
```

The Jetson config (`config.jetson.yaml`) differs from the default in three ways:
1. `config_device: cuda` — YOLO runs on GPU via TRT FP16 engine
2. `config_vlm_device: cpu` — VLM runs on CPU (GPU memory is reserved for YOLO)
3. `config_yolo_model: yolov11v28_jingtao.engine` — pre-built TRT engine

## Config Notes

A few config values are especially important for the optional VLM path:

- `config_frame_resolution`: the input layer resizes raw frames once here, and downstream cropper logic works in that normalized frame space.
- Once ROI is locked, the current YOLO layer now uses the native `roi_layer_image` crop shape for inference instead of forcing the crop back to the model's default square `imgsz`. ROI still operates inside the normalized frame space produced by `config_frame_resolution`, but YOLO no longer treats ROI crops like full-frame inputs at inference size.
- `config_vlm_enabled`: enables the real VLM inference path, but cropper visualization can still run without VLM inference enabled.
- `config_vlm_model`: selects the VLM model when `config_vlm_enabled` is true.
- `config_vlm_device`: overrides `config_device` for the VLM only. Empty string inherits `config_device`. Use `cpu` on Jetson to reserve GPU for YOLO.
- `config_yolo_imgsz`: optional `[H, W]` pair to force a specific inference resolution for TRT engines compiled at non-square shapes. Null means use the model default (640).
- `config_vlm_crop_feedback_enabled`: when `true`, VLM may request a new crop round; when `false`, the first dispatched crop completes that track after JSON classification.
- `config_vlm_crop_cache_size`: number of crops collected in the current round before cropper scores that round and sends one best image to VLM. After an accepted result is written into persistent state, tracking can continue but cropper and VLM are skipped for that track.
- `config_vlm_dead_after_lost_frames`: number of consecutive `lost` tracking updates before the cropper marks the track `dead`. Once dead, a never-filled first cache round may still dispatch its best available crop instead of being dropped.

## How To Use This Folder

1. Start with `pipeline_layers_and_interactions.md` when deciding whether a layer interface is correct.
2. Use the layer-local README files for runnable examples and practical commands.
3. For current YOLO detector tag policy, read `src/yolo-layer/TAG_FILTER_BEHAVIOR.md`. That document explains which detector labels are forwarded downstream right now and notes that editing `src/yolo-layer/class_map.py` changes the Python pipeline's detector behavior.
4. Treat helper scripts as orchestration utilities that must stay compatible with the pipeline contract.
## VLM Ack Loop

The current branch also implements a stricter optional VLM dispatch loop around the documented cropper and state layers:

- one track is sent once after its crop cache fills
- the same track is not resent unless VLM explicitly returns a retry acknowledgement
- if retry is requested while the truck remains visible, the cropper waits for a newer or better candidate
- if retry is requested after the truck leaves scope, the cropper sends one final best-available candidate and the state layer records that finalization
- if a track stays `lost` for `config_vlm_dead_after_lost_frames`, the cropper marks it `dead`; if that track never filled its first cache round, the cropper now still sends the best available partial-cache crop


## VLM Semantic Notes

`pipeline_layers_and_interactions.md` now documents a narrower current VLM contract.

Short version:

- VLM first decides whether the crop is one of the currently active YOLO labels, such as `truck` or `bus`
- if yes, it returns a small rigid JSON payload
- the current JSON focuses on:
  - `wheel_count`
  - `ack_status`
  - `retry_reasons`
- the current debug-image workflow in `src/vlm-layer` saves:
  - `track id`
  - `model id`
  - prompt
  - actual VLM output
- if no, the track is acknowledged with reason `no`
- subtype-style free-form semantic notes are not part of the current contract

Once a track is accepted and its semantic JSON is written into persistent state, that track is treated as progressed: YOLO and tracking may still keep following it, but cropper and VLM no longer run again for that same track unless the contract is explicitly changed.

`config_vlm_crop_feedback_enabled=false` enables a single-shot classify-and-complete mode: after one full crop round and one dispatch, VLM returns JSON classification, the track is marked progressed in persistent state, and cropper/VLM stop running for that track while tracking may continue.

The current target-label gate also adds a terminal split:

- VLM rejection of the currently flagged detector labels means the track is marked `no`
- lost-threshold expiry before a successful semantic result means the track is marked `dead`
- accepted target-label JSON means the track is marked `done`

## Changes

## 2026-04-16 (Jetson optimization)

- Added `config_vlm_device` config key: per-component device override for VLM; empty = inherit `config_device`.
- Added `config_yolo_imgsz` config key: explicit `[H, W]` inference size for non-square TRT engines.
- Fixed `benchmark.py` double YOLO per frame: `update_roi_state` is a no-op post-lock; single YOLO call now feeds both the pipeline and ROI state (+49% pipeline FPS).
- `src/yolo-layer/detector.py`: YOLO warmup before VLM load (NvMap reservation), FP16 inference, `is_trt` flag for engine-aware dispatch.
- `src/tracking-layer/tracker.py`: `build_tracking_layer_package` emits flat arrays alongside nested tracks.
- See `JETSON_OPTIMIZATION.md` for full session record and benchmark results.

## 2026-04-15

- Added `pipeline/benchmark.py` (moved from `src/evaluation-output-layer/benchmark.py`): video-only end-to-end profiler that reads `src/configuration-layer/config.yaml`. ROI-specific matrix helpers live under `src/roi-layer/test/`.

## 2026-04-16

- Updated `benchmark.py` path setup so `config_vlm_runtime_mode=async` can import `AsyncVLMWorker` from `src/vlm-layer/util/visualize_vlm_realtime.py` instead of silently dropping to inline mode because of a local import failure.
- Added `src/roi-layer/util/visualize_roi_vlm_upson.py` as the clip-specific Upson ROI/VLM helper for the `0:48` to `1:48` demo window.
- Updated the pipeline-level helper list and detector-policy wording so the docs no longer imply the active YOLO filter is always COCO-style.

## 2026-04-11

- Updated pipeline docs to match the simplified active VLM contract used in
  this session.
- Replaced the older truck-subtype-focused description with the current
  narrower target-label gate:
  first decide whether the crop matches an active YOLO label such as `truck`
  or `bus`, then if yes return rigid JSON.
- Documented that the active VLM JSON now focuses on:
  - `wheel_count`
  - `ack_status`
  - `retry_reasons`
- Documented that `vlm_image_quality_notes` is no longer part of the active
  VLM contract.
- Clarified that `vehicle_state_layer_truck_type` is now a legacy
  compatibility slot relative to the simplified current VLM prompt.
- Documented the real config-driven VLM runner and the simplified saved
  debug-image outputs in the VLM-layer docs.

## 2026-04-10

- Added `src/yolo-layer/TAG_FILTER_BEHAVIOR.md` as the explicit reference for which YOLO/COCO tags are currently forwarded into the pipeline versus discarded by the repo's class filter.
- Updated `pipeline_layers_and_interactions.md` to clarify that the bundled detector may know many labels, but downstream behavior depends on the active `class_map.py` filter.
- Updated `pipeline/README.md` to point readers to the new YOLO tag-policy document when they need to understand or modify detector behavior.
- Added pipeline-level documentation for `config_vlm_dead_after_lost_frames`, dead-track partial-cache dispatch, and the current terminal split where VLM marks tracks `dead` when `is_truck=false` and `done` when truck semantics are accepted.
- Refined terminal semantics so VLM rejection of non-flagged labels is now
  tracked as `no`, while `dead` remains reserved for the cropper lost-threshold
  path.
- Documented `src/roi-layer/util/visualize_roi_vlm.py` as the helper that first
  calibrates ROI from full-frame detections and then runs YOLO + VLM only
  inside the locked ROI crop.
- Documented `src/vlm-layer/util/visualize_vlm_roi.py` as the helper that first
  calibrates ROI and then runs the full tracking -> cropper selection -> VLM
  loop inside the locked ROI crop.
- Documented `src/vlm-layer/util/visualize_vlm_roi_realtime.py` as the async
  ROI-integrated helper that keeps the feed moving after ROI lock while VLM
  runs in the background.
- Documented that once ROI is locked, the current YOLO layer now uses the
  native ROI crop shape for inference instead of forcing ROI inputs back to the
  model's default square inference size.
- Cleaned up ROI inference sizing so the YOLO layer now rounds ROI crop shapes
  to stride-safe values before inference, avoiding repetitive Ultralytics
  warning spam without changing the intended ROI compute path.

## 2026-04-09

- Documented `src/vlm-layer/util/visualize_vlm.py` in `pipeline/README.md` as the end-to-end VLM path visualizer.
- Clarified `config_vlm_crop_cache_size` in `pipeline/README.md`: cache retention follows score-based top-K (per pipeline scoring/selection), not newest-only FIFO.
- Noted the optional runnable path through the cropper and VLM when `config_vlm_enabled` is true.
- Fixed `codex_ground_rules.md` to reference `pipeline_layers_and_interactions.md` instead of the non-existent `interactions.md`.

## 2026-04-08

- Expanded `pipeline/README.md` so it now points to the contract document, explains the currently runnable layer path, and links the practical helper scripts that were added around the documented pipeline interfaces.
- Added pipeline-level truck-type semantics documentation so `vehicle_state_layer_truck_type` is no longer just a field name but a documented subtype contract with a recommended v1 vocabulary.
- Added `config_vlm_crop_feedback_enabled` to the pipeline contract so the optional VLM path can operate either as a retry-capable feedback loop or as single-shot classify-and-complete.
- Removed the generic retry reason `other` from the VLM feedback contract so retry requests now use only explicit machine-readable causes.
- Narrowed the VLM retry vocabulary further so retry requests may only use `occluded` or `bad_angle`.
