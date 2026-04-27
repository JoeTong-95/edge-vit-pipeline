# Review Package Spec

This document locks the implementation details for the review package, human-truth workflow, and report evaluation flow.

It is meant for the next coding agent to execute directly.

## 1. Goal

Build one repeatable local workflow that does three things:

1. Run the headless pipeline on a video and save review artifacts locally on Jetson.
2. Let humans label those artifacts into a SQLite truth database.
3. Compare tracker and VLM outputs against that truth database for report metrics.

Accuracy review and speed benchmarking are separate workflows.

## 2. Baseline Assumptions

- Use 2 evaluation clips:
  - one day clip
  - one night clip
- Baseline report config is locked before wide comparisons:
  - one YOLO TRT model
- one VLM model
  - one VLM runtime mode
- Local recorded video on Jetson is the baseline input for report comparisons.

Speed benchmarking note for the current branch:

- accuracy review and baseline report runs still lock one VLM model at a time
- separate speed benchmarking for this stage should compare all three modular VLM options:
  - `smolvlm_256m`
  - `qwen_0_8b`
  - `gemma_e2b_local`
- for each of those, compare `cpu` vs `cuda`
- do not require INT8 / low-bit VLM benchmarking for this report stage because the repo does not yet provide one clean, modular, apples-to-apples INT8 path across all three models
- record quantization as a later experiment after the modular cpu/cuda path is implemented and stable

## 3. Current Pipeline Reality

The current pipeline already provides these useful pieces:

- tracking IDs and per-frame tracking rows
- VLM crop cache and dispatch metadata
- VLM normalized output
- VLM ack status and retry reasons
- JSONL output logging in `initialize_pipeline.py`

Important current fields already available:

- tracker:
  - `track_id`
  - `bbox`
  - `detector_class`
  - `confidence`
  - `status`
- VLM cropper:
  - `vlm_dispatch_mode`
  - `vlm_dispatch_reason`
  - `vlm_dispatch_cached_crop_count`
- VLM result:
  - `vlm_ack_status`
  - `vlm_retry_reasons`
  - current normalized `is_target_vehicle`

Implementation note:

- internally the VLM layer should use `is_target_vehicle`
- for review/export, add `is_target_vehicle`
- `is_target_vehicle` means whether VLM agrees that the crop belongs to the active target-vehicle set for that backend

## 4. Top-Level Folder Contract

Create a repo-top-level folder:

- `review-package/`

Each review run should create a run folder:

- `review-package/runs/<run_id>/`

Recommended `run_id` format:

- `<YYYYMMDD_HHMMSS>_<video_stem>_<config_tag>`

Each run folder should contain:

- `new_tracks/`
- `vlm_accepted_targets/`
- `metadata/`
- `summaries/`
- `artifacts/`

The shared truth DB should live at:

- `review-package/human_truth.sqlite`

## 5. Folder Contents

### `new_tracks/`

Purpose:

- human truth for unique tracked targets in the video

Save rule:

- save one representative image per new track ID
- only save for downstream target detector classes:
  - `pickup`
  - `van`
  - `truck`
  - `bus`

Representative frame policy:

- default rule: save the first frame where tracking status is `new`
- also record the tracker bbox and confidence from that frame

Reason:

- this is deterministic
- easy to implement now
- good enough for the first human-truth pass

Future improvement if needed:

- optionally add a "best representative crop" mode later

Filename format:

- `track_<track_id>__frame_<frame_index>__class_<target_class>.jpg`

### `vlm_accepted_targets/`

Purpose:

- review what was actually accepted and sent to VLM

Save rule:

- save one crop per dispatch event that actually goes to VLM
- save the exact crop image passed downstream by the cropper

Filename format:

- `track_<track_id>__frame_<frame_index>__dispatch_<dispatch_mode>__class_<target_class>.jpg`

### `metadata/`

Files:

- `new_tracks.csv`
- `vlm_accepted_targets.csv`
- `run_events.jsonl`

`new_tracks.csv` should have one row per saved `new_tracks` image.

`vlm_accepted_targets.csv` should have one row per saved `vlm_accepted_targets` image.

`run_events.jsonl` should be a richer event stream for debugging and replay if needed.

### `summaries/`

Files:

- `run_summary.json`
- `run_summary.md`

### `artifacts/`

Purpose:

- optional larger debug assets

Examples:

- raw VLM text dumps when enabled
- copied config snapshot
- benchmark output
- spill queue files when a run explicitly chooses to preserve them

## 6. Metadata Schema

### `new_tracks.csv`

Required columns:

- `run_id`
- `source_video`
- `frame_index`
- `track_id`
- `tracker_status`
- `target_class`
- `bbox_x1`
- `bbox_y1`
- `bbox_x2`
- `bbox_y2`
- `tracker_confidence`
- `image_relpath`
- `review_status`

`review_status` starts empty and is filled by the frontend only if useful for caching UI state.

### `vlm_accepted_targets.csv`

Required columns:

- `run_id`
- `source_video`
- `frame_index`
- `track_id`
- `target_class`
- `bbox_x1`
- `bbox_y1`
- `bbox_x2`
- `bbox_y2`
- `tracker_confidence`
- `dispatch_mode`
- `dispatch_reason`
- `dispatch_cached_crop_count`
- `from_cache`
- `model_id`
- `query_type`
- `is_target_vehicle`
- `ack_status`
- `retry_reasons`
- `image_relpath`
- `raw_event_relpath`

Rules:

- `from_cache` is a boolean export field
- set `from_cache=true` for dispatches that depend on the crop cache as opposed to immediate single-frame behavior
- since the current VLM cropper is cache-based by design, this field should still be exported because future comparisons may include reduced-cache or no-cache variants

### `run_events.jsonl`

This is the richer machine-readable stream.

Required event types:

- `new_track_saved`
- `vlm_dispatch_saved`
- `vlm_result_logged`
- `run_summary`

Each event should include:

- `event_type`
- `run_id`
- `source_video`
- `frame_index`
- `track_id` when applicable
- `timestamp_utc`

## 7. Minimal Review Payload

This is the minimal result payload to expose in review/export outputs:

- `track_id`
- `frame_index`
- `source_video`
- `model_id`
- `query_type`
- `is_target_vehicle`
- `ack_status`
- `retry_reasons`
- `target_class`

Field definitions:

- `target_class`
  - normalized downstream class
  - one of `pickup`, `van`, `truck`, `bus`
- `is_target_vehicle`
  - whether VLM agrees this item is in the active target-vehicle set for the configured backend
- `ack_status`
  - current VLM ack status from pipeline normalization
- `retry_reasons`
  - short list from pipeline normalization

Do not include free-form `decision` or `notes` fields in the default structured payload.

Raw model text:

- keep only as optional debug output
- not part of the main review payload

## 8. Human Truth Database Schema

SQLite file:

- `review-package/human_truth.sqlite`

Tables:

### `review_items`

One row per image shown to humans.

Columns:

- `id` INTEGER PRIMARY KEY
- `run_id` TEXT NOT NULL
- `item_type` TEXT NOT NULL
- `source_video` TEXT NOT NULL
- `frame_index` INTEGER NOT NULL
- `track_id` TEXT NOT NULL
- `target_class` TEXT
- `image_relpath` TEXT NOT NULL
- `metadata_relpath` TEXT
- `created_at_utc` TEXT NOT NULL

`item_type` values:

- `new_track`
- `vlm_accepted_target`

### `review_labels`

One row per human label action.

Columns:

- `id` INTEGER PRIMARY KEY
- `review_item_id` INTEGER NOT NULL
- `reviewer` TEXT
- `label_key` TEXT NOT NULL
- `label_value` TEXT
- `created_at_utc` TEXT NOT NULL

Allowed `label_key` values:

- `wrong_class`
- `true_class`
- `repeat`
- `bad_crop`

### `review_highlights`

One row per metadata highlight / comment.

Columns:

- `id` INTEGER PRIMARY KEY
- `review_item_id` INTEGER NOT NULL
- `reviewer` TEXT
- `field_name` TEXT
- `highlight_text` TEXT
- `comment_text` TEXT
- `created_at_utc` TEXT NOT NULL

Purpose:

- support double-click highlighting of suspicious metadata
- keep image review and metadata comments linked together

## 9. Review Frontend Contract

Build:

- `pipeline/review_app.py`

Requirements:

- lightweight local app
- optimized for SSH + local browser / forwarded browser use
- image shown beside metadata
- keyboard-first review flow

Minimum shortcut mapping:

- `q` -> `wrong_class`
- `w` -> `true_class`
- `e` -> `repeat`
- `r` -> `bad_crop`

UI behavior:

- show next unlabeled item quickly
- support filtering by:
  - run
  - item type
  - target class
  - source video
- metadata panel should support double-click highlight of suspicious fields/text
- highlight action should open a small comment box and save to `review_highlights`

## 10. Evaluation Logic

Build:

- `pipeline/compare_against_human_truth.py`

This script should not ask humans to do anything new.

It should only compare:

- tracker / VLM outputs from the original run
- against labels and comments already saved in `human_truth.sqlite`

### Tracker truth evaluation

Goal:

- determine which tracked targets are real trucks in the video

Use:

- `new_tracks`
- human labels on those items

Report:

- `total_tracked_targets`
- `total_human_confirmed_trucks`
- `total_repeat_tracks`
- `tracker_precision_like`

### VLM agreement evaluation

Goal:

- compare VLM result against human truth for matching track IDs

Report:

- `total_vlm_accepted_targets`
- `vlm_type_agreement_rate`
- `vlm_class_agreement_rate`
- `vlm_retry_rate`
- `vlm_false_accept_count`
- `vlm_false_reject_count`

### Metadata quality evaluation

Goal:

- quantify problematic metadata for IDs that have VLM outputs

Report:

- `matched_id_count`
- `problematic_metadata_count`
- `problematic_metadata_rate`
- counts grouped by highlighted field name if available

## 11. Overflow / Cache Evaluation Policy

Overflow / cache behavior must be measurable in the saved outputs.

Implementation rule:

- every `vlm_accepted_targets` row must include:
  - `dispatch_mode`
  - `dispatch_reason`
  - `dispatch_cached_crop_count`
  - `from_cache`

Reason:

- this allows later comparison between:
  - normal cached behavior
  - reduced cache settings
  - spill / overflow behavior
- report analysis can then estimate what realtime-no-cache would miss

Do not create a separate cache folder first.

Preferred first implementation:

- keep one artifact structure
- encode cache behavior in metadata fields

## 12. Performance Workflow

Build:

- `pipeline/run_experiment_matrix.py`
- `pipeline/build_report_summary.py`

Performance workflow is separate from truth review.

Performance questions:

- how fast does the pipeline run
- how does speed change by VLM backend / device / runtime

Do not mix human-truth review into performance runs.

Primary speed axes:

- VLM backend
- VLM device
- VLM runtime mode

Current Jetson safety note:

- do not treat every `backend × device` case as equally safe to batch-run unattended
- on this machine, attempting the unstable full three-backend `cpu/cuda` matrix has been associated with:
  - SSH disconnect
  - temporary ping failure
  - host recovery only after about `5` minutes
- the matrix runner should therefore default to a safe mode that skips currently known risky combinations unless the operator explicitly opts in

Secondary axes if time remains:

- YOLO model
- source resolution
- ROI on/off
- cache size tuning

Metrics:

- `fps`
- `ms_per_frame`
- `yolo_ms`
- `tracking_ms`
- `state_ms`
- `vlm_query_ms`
- `vlm_queue_depth`
- `vlm_completed_count`
- `detections_per_frame`
- `tracks_per_frame`

## 13. Script Build Order

1. Modularize VLM backend selection
2. Build `pipeline/run_deployment_review.py`
3. Build `review-package/` folder creation and metadata writers
4. Build `pipeline/review_app.py`
5. Build `human_truth.sqlite` schema init
6. Build `pipeline/compare_against_human_truth.py`
7. Lock baseline config
8. Run day + night baseline
9. Build `pipeline/run_experiment_matrix.py`
10. Build report summary generation

## 14. Recommended First Implementation Choices

To reduce ambiguity for the next agent, these choices are locked in:

- `new_tracks` uses first `status == new` frame, not "best crop"
- `vlm_accepted_targets` saves one image per actual VLM dispatch event
- `is_target_vehicle` is the export/review field for the active target gate
- cache behavior is tracked with metadata fields, not separate folders
- raw VLM text is optional debug only
- review labels are limited to:
  - `wrong_class`
  - `true_class`
  - `repeat`
  - `bad_crop`
- performance workflow is speed-only
- truth workflow owns accuracy evaluation
- the experiment-matrix helper should be conservative by default on Jetson and require an explicit opt-in before re-running host-risky backend/device combinations
