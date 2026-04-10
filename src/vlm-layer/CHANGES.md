Layer changes in this branch

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
