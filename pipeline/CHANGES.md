# Pipeline Changes

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
