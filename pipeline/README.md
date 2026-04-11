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

- `src/yolo-layer/visualize_yolo.py`: config-driven YOLO-only visualization with optional live preview, optional video output, and optional SQLite metrics.
- `src/tracking-layer/visualize_tracking.py`: config-driven YOLO plus tracking visualization with optional live preview, optional video output, and optional SQLite metrics.
- `src/roi-layer/visualize_roi.py`: config-driven ROI discovery visualization with optional live preview and optional video output.
- `src/roi-layer/visualize_roi_vlm.py`: calibrates ROI from full-frame detections first, then after ROI lock runs YOLO and VLM only inside the ROI crop and shows the selected ROI-local crop sent to VLM.
- `src/vlm-frame-cropper-layer/visualize_vlm_frame_cropper.py`: config-driven crop-cache visualization showing per-track crop history and the current crop selected for VLM input.
- `src/vlm-layer/visualize_vlm.py`: end-to-end orchestration visualizer (input → YOLO → tracking → cropper cache/dispatch → VLM inference) with structured prompt preview, raw response, and ack/retry logging; reads `config.yaml` like the other helpers.
- `src/vlm-layer/visualize_vlm_roi.py`: calibrates ROI first, then runs the downstream tracking -> cropper cache/selection -> VLM loop only inside the locked ROI crop.
- `src/vlm-layer/visualize_vlm_realtime.py`: non-blocking VLM helper that keeps the feed running at source FPS while VLM inference happens on a background worker, making it easier to see backlog and whether inference keeps up with a live-like stream.
- `src/vlm-layer/visualize_vlm_roi_realtime.py`: ROI-integrated non-blocking VLM helper; calibrates ROI first, then keeps the ROI-local feed moving while cropper-selected VLM inference runs on a background worker.
- `src/tracking-layer/automated_evaluation.py`: sequential benchmark sweep across CPU/CUDA, YOLO v8/v10/v11, and tracking on/off.
- `src/tracking-layer/plot_evaluation_results.py`: creates styled summary plots from the evaluation SQLite output.

## Config Notes

A few config values are especially important for the optional VLM path:

- `config_frame_resolution`: the input layer resizes raw frames once here, and downstream cropper logic works in that normalized frame space.
- Once ROI is locked, the current YOLO layer now uses the native `roi_layer_image` crop shape for inference instead of forcing the crop back to the model's default square `imgsz`. ROI still operates inside the normalized frame space produced by `config_frame_resolution`, but YOLO no longer treats ROI crops like full-frame inputs at inference size.
- `config_vlm_enabled`: enables the real VLM inference path, but cropper visualization can still run without VLM inference enabled.
- `config_vlm_model`: selects the VLM model when `config_vlm_enabled` is true.
- `config_vlm_crop_feedback_enabled`: when `true`, VLM may request a new crop round; when `false`, the first dispatched crop completes that track after JSON classification.
- `config_vlm_crop_cache_size`: number of crops collected in the current round before cropper scores that round and sends one best image to VLM. After an accepted result is written into persistent state, tracking can continue but cropper and VLM are skipped for that track.
- `config_vlm_dead_after_lost_frames`: number of consecutive `lost` tracking updates before the cropper marks the track `dead`. Once dead, a never-filled first cache round may still dispatch its best available crop instead of being dropped.

## How To Use This Folder

1. Start with `pipeline_layers_and_interactions.md` when deciding whether a layer interface is correct.
2. Use the layer-local README files for runnable examples and practical commands.
3. For current YOLO detector tag policy, read `src/yolo-layer/TAG_FILTER_BEHAVIOR.md`. That document explains which COCO labels are forwarded downstream right now and notes that editing `src/yolo-layer/class_map.py` changes the Python pipeline's detector behavior.
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
  - `estimated_weight_kg`
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
