Layer changes in this branch

- Simplified the active VLM prompt contract to a short rigid JSON-only format:
  first answer whether the crop matches an active YOLO label such as `truck`
  or `bus`, then if yes return only numeric `wheel_count`,
  numeric `estimated_weight_kg`, `ack_status`, and `retry_reasons`.
- Removed `vlm_image_quality_notes` from the active VLM contract and kept
  `retry_reasons` as the only structured explanation for bad crops such as
  `occluded` or `bad_angle`.
- Stopped asking the model for `truck_type` and prompt-side `confidence` in the
  active contract.
- Added `run_config_vlm_once.py` as the real config-driven runner for:
  `input -> YOLO -> tracking -> cropper -> VLM`.
- Added saved debug-image generation for both sample outputs and real
  config-driven VLM runs.
- Iterated the saved debug-image layout during this session:
  dark theme, larger fonts, top-row crop preview, and simplified displayed
  fields (`track id`, `model id`, prompt, actual VLM output).
- Updated parsing defaults so numeric fields now fall back to `0` instead of
  string `unknown` in the active prompt/JSON path.

- Added `VLMAckPackage`, `build_vlm_ack_package`, and `serialize_vlm_ack_package` so the VLM layer can explicitly acknowledge whether a dispatched crop was accepted, needs retry, or should be finalized with the current best crop.
- Documented the VLM layer's acknowledgement role in the layer README so it is clear how this layer now participates in the one-shot dispatch loop.
- Added `preview_vlm_applied_prompt` for visualization/debug of the processor chat-template string.
- Added `visualize_vlm.py`: three-column view (raw+tracks, cropper cache/selection, dispatch/VLM prompt/response/ack log) wired to real collect→dispatch→infer→ack loop.
- Updated `README.md` with relative links to pipeline docs, `visualize_vlm.py` usage, `preview_vlm_applied_prompt`, and orchestration scope notes.
- Raised minimum `transformers` to **5.x** for Qwen3.5 (`qwen3_5`): `initialize_vlm_layer` now fails fast with an upgrade hint on 4.x, and docker requirement pins use `transformers>=5.0.0,<6.0.0`.
- `visualize_vlm.py`: tighter crop-cache column width via `crop_cache_panel_width_unscaled`, default wider dispatch column (640px), `--right-panel-width`, and `np.hstack` layout so columns sit flush without dead space.

- Added a VLM loop visualizer view that shows the selected crop, the last image actually sent to VLM, the acknowledgement decision, and whether metadata was accepted or a retry round is being collected.
- Moved accept-vs-retry judgement into the VLM output contract itself: `vehicle_semantics_v1` now returns structured JSON with `ack_status`, retry reasons like `occluded` or `bad_angle`, and the semantic classification fields when the image is good enough.
- Added config-driven single-shot mode via `config_vlm_crop_feedback_enabled=false`, so one dispatched crop can produce final JSON classification and permanently skip further cropper/VLM work for that track.
- Removed the catch-all retry reason `other` from the VLM schema, parser fallbacks, and visualizer summaries so retry messages stay explicit and actionable.
- Narrowed retry reasons to only `occluded` or `bad_angle` across the prompt, parser, and visualizer fallbacks.
- Added `visualize_vlm_realtime.py`, a non-blocking helper that keeps the feed moving while VLM inference runs on a background worker so queue lag and throughput can be inspected.

## 2026-04-10

- Added explicit truck gate semantics to the VLM normalization and ack path:
  `is_truck=false` now returns an accepted `not_truck` acknowledgement so
  downstream state can mark the track `dead`.
- Updated the visualizer orchestration so cropper dispatch mode
  `dead_best_available` uses a single-shot truck check and still produces a
  final VLM decision for incomplete cache rounds that ended in a dead track.
- Updated VLM visual debug state so terminal tracks are shown as `dead` or
  `done` instead of the older generic progressed wording.
- Updated the rejection path to use acknowledgement reason `no`, so VLM
  rejection of the currently flagged labels is distinct from cropper-side
  `dead` caused by the lost threshold.
- Added `visualize_vlm_roi.py`, a cross-layer helper that shows ROI
  calibration first and then runs the tracking -> cropper selection -> VLM
  sequence inside the locked ROI crop.
- Added `visualize_vlm_roi_realtime.py`, an async ROI-integrated helper that
  keeps the feed moving after ROI lock while VLM inference runs in a background
  worker.

## 2026-04-15

- Added true batched VLM inference helpers: `infer_vlm_semantics_batch` and
  `run_vlm_inference_batch`, enabling multi-crop micro-batching via a single
  `model.generate(...)` call.
- Updated `visualize_vlm_realtime.py` async VLM worker to support micro-batching
  (`--vlm-batch-size`, `--vlm-batch-wait-ms`) and optional overflow
  spill-to-disk queue (`--vlm-spill-queue`) for a cache-and-run workflow.
- Added `vlm_deferred_queue.py` and `run_deferred_vlm_queue.py` to persist crop
  tasks as JSONL (PNG base64) and process them offline later.
- Added `VLM_OPTIMIZATION_NOTES.md` documenting realtime vs cache-and-run modes,
  and how to run both.
