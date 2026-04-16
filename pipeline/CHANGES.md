# Pipeline Changes

## 2026-04-15

- Added `pipeline/benchmark.py` (moved from `src/evaluation-output-layer/benchmark.py`): video-only end-to-end profiler that reads `src/configuration-layer/config.yaml`. ROI-specific matrix helpers live under `src/roi-layer/`.

## 2026-04-11

- Updated pipeline docs to match the simplified active VLM contract used in
  this session.
- Replaced the older truck-subtype-focused description with the current
  narrower target-label gate:
  first decide whether the crop matches an active YOLO label such as `truck`
  or `bus`, then if yes return rigid JSON.
- Documented that the active VLM JSON now focuses on:
  - `wheel_count`
  - `estimated_weight_kg`
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
- Documented `src/roi-layer/visualize_roi_vlm.py` as the helper that first
  calibrates ROI from full-frame detections and then runs YOLO + VLM only
  inside the locked ROI crop.
- Documented `src/vlm-layer/visualize_vlm_roi.py` as the helper that first
  calibrates ROI and then runs the full tracking -> cropper selection -> VLM
  loop inside the locked ROI crop.
- Documented `src/vlm-layer/visualize_vlm_roi_realtime.py` as the async
  ROI-integrated helper that keeps the feed moving after ROI lock while VLM
  runs in the background.
- Documented that once ROI is locked, the current YOLO layer now uses the
  native ROI crop shape for inference instead of forcing ROI inputs back to the
  model's default square inference size.
- Cleaned up ROI inference sizing so the YOLO layer now rounds ROI crop shapes
  to stride-safe values before inference, avoiding repetitive Ultralytics
  warning spam without changing the intended ROI compute path.

## 2026-04-09

- Documented `src/vlm-layer/visualize_vlm.py` in `pipeline/README.md` as the end-to-end VLM path visualizer.
- Clarified `config_vlm_crop_cache_size` in `pipeline/README.md`: cache retention follows score-based top-K (per pipeline scoring/selection), not newest-only FIFO.
- Noted the optional runnable path through the cropper and VLM when `config_vlm_enabled` is true.
- Fixed `codex_ground_rules.md` to reference `pipeline_layers_and_interactions.md` instead of the non-existent `interactions.md`.

## 2026-04-08

- Expanded `pipeline/README.md` so it now points to the contract document, explains the currently runnable layer path, and links the practical helper scripts that were added around the documented pipeline interfaces.
- Added pipeline-level truck-type semantics documentation so `vehicle_state_layer_truck_type` is no longer just a field name but a documented subtype contract with a recommended v1 vocabulary.
- Added `config_vlm_crop_feedback_enabled` to the pipeline contract so the optional VLM path can operate either as a retry-capable feedback loop or as single-shot classify-and-complete.
- Removed the generic retry reason `other` from the VLM feedback contract so retry requests now use only explicit machine-readable causes.
- Narrowed the VLM retry vocabulary further so retry requests may only use `occluded` or `bad_angle`.
