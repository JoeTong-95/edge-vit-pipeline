# May Report Todo v2

## Scope

- Use 2 deployment clips for report testing:
  - 1 daytime
  - 1 nighttime
- Save all review artifacts locally on Jetson storage
- Add one top-level repo folder for review / experiment outputs so teammates can SSH in and help label ground truth

Suggested top-level folder:

- `review-package/`
- detailed implementation contract:
  - `may-report-package/review_package_spec.md`

## Baseline Rule

Before broad comparisons, lock one report baseline:

- one YOLO TRT model
- one VLM model
- one runtime mode
- two evaluation clips:
  - day
  - night

All later comparisons should be measured against that locked baseline.

---

## Priority 1: Modular VLM Layer

Goal:

- make VLM the main swap-in / swap-out layer for experiments

Configs to support first:

- backend:
  - `smolvlm_256m`
  - `qwen_0_8b`
  - `gemini_e2b`
- device:
  - `cpu`
  - `cuda`
- runtime:
  - `inline`
  - `async`

Deliverables:

- stable backend interface in `src/vlm-layer`
- config keys for backend / device / runtime
- one normalized VLM output schema for downstream layers

---

## Priority 2: Human Truth Review Package

Goal:

- generate local review artifacts that humans can label quickly

Top-level output folder:

- `review-package/`

Contents to generate:

- `review-package/new_tracks/`
  - one saved image per new tracked target object
- `review-package/vlm_accepted_targets/`
  - one saved crop per target object that is actually accepted and sent to VLM
- `review-package/metadata/`
  - readable CSV / JSON aligned to image filename and track id
- `review-package/summaries/`
  - per-run summary JSON / Markdown
- `review-package/human_truth.sqlite`
  - local human-truth database

Human labeling fields:

- `wrong_class`
- `true_class`
- `repeat`
- `bad_crop`

Frontend direction:

- lightweight local marker frontend
- image on one side, metadata on the other
- keyboard-first labeling with shortcuts such as `q`, `w`, `e`, `r`
- minimal actions only:
  - `wrong_class`
  - `true_class`
  - `repeat`
  - `bad_crop`
- metadata viewer should support comment-style highlight markup:
  - operator can double-click suspicious text / fields while the linked image is shown
  - save those highlights with the review record for later analysis

Minimal structured VLM output target:

- keep only fields needed for review and comparison
- align with the current pipeline's existing normalized output and sample JSON
- suggested per-result schema:
  - `track_id`
  - `frame_index`
  - `source_video`
  - `model_id`
  - `query_type`
  - `is_type`
  - `ack_status`
  - `retry_reasons`
  - `target_class`

Where:

- `is_type` means whether the VLM agrees with the YOLO target tag / type
- `retry_reasons` should stay as a short list because it is already part of the ack flow
- `target_class` is the final normalized downstream class we care about:
  - `pickup`
  - `van`
  - `truck`
  - `bus`
- avoid extra prose fields such as `decision` or `notes` in the main result payload
- if needed, keep raw model text only as an optional debug artifact, not the default review payload

---

## Priority 3: Truth Database And Evaluation

Goal:

- save human review once, then evaluate tracker / VLM outputs against that database

Purpose of the human-truth package:

- use tracking to build truth for what trucks actually appear in a video
- compare that against:
  - what VLM says is truck or not
  - the final VLM output JSON
- use metadata review to measure the percentage of problematic metadata for IDs that match VLM output

Overflow cache handling:

- treat overflow / spill behavior as a traceable evaluation axis
- either:
  - give cached results their own folder
  - or mark each saved artifact / row with a `from_cache` flag
- this should let us quantify how much realtime-no-cache misses compared with cached processing

Evaluation flow:

- human review interface writes labels and comments into `human_truth.sqlite`
- `compare_against_human_truth.py` should compare reviewed truth against original tracker / VLM outputs
- no separate manual "accuracy experiment" workflow is needed beyond running the review package and then checking against the database

Outputs needed:

- tracker-truth summary:
  - total tracked targets
  - total human-confirmed trucks
  - tracker duplicate / repeat cases
- VLM agreement summary:
  - total VLM accepted targets
  - truck / not-truck agreement against human truth
  - class agreement against human truth
- metadata quality summary:
  - percentage of problematic metadata among matched IDs
  - highlighted suspicious fields / comments from reviewers

---

## Priority 4: Performance Experiments

Goal:

- compare headless pipeline speed only

Note:

- performance testing does not need accuracy review in the loop
- this section is only about how fast the pipeline runs under different configs

Main variables to test first:

- VLM backend
- VLM device
- VLM runtime mode

If time allows:

- YOLO model swap
- source resolution
- ROI on / off
- cache size tuning

Metrics to record:

- end-to-end FPS
- end-to-end ms/frame
- YOLO ms/frame
- tracking ms/frame
- state ms/frame
- VLM query time
- VLM queue / drain behavior
- detections per frame
- tracks per frame

---

## Input Strategy

Baseline for comparison:

- use local recorded video files on Jetson

Reason:

- more repeatable
- avoids adding network / stream variability

Optional later test:

- stream video from another device for deployment realism

Do not use streaming as the primary baseline unless needed.

---

## Scripts To Build

- `pipeline/run_deployment_review.py`
  - runs full headless pipeline
  - writes review artifacts into `review-package/`

- `pipeline/review_app.py`
  - lightweight local frontend for marking ground truth

- `pipeline/compare_against_human_truth.py`
  - computes accuracy / coverage metrics from reviewed outputs

- `pipeline/run_experiment_matrix.py`
  - runs multiple config combinations and saves comparison tables

- `pipeline/build_report_summary.py`
  - generates report-ready CSV / JSON / Markdown summaries

---

## Recommended Build Order

1. Modularize VLM backends
2. Build `review-package/` artifact generator
3. Build lightweight review frontend + `human_truth.sqlite`
4. Lock one baseline production config
5. Run the baseline on 2 clips: day + night
6. Compare only the most realistic VLM alternatives
7. Do YOLO / resolution sweeps only if time remains
8. Generate report-ready comparison outputs

---

## Immediate Next Tasks

- [ ] Add `review-package/` top-level folder convention
- [ ] Modularize VLM backend selection
- [ ] Build headless review artifact generator
- [ ] Save new-track images
- [ ] Save accepted target crops
- [ ] Save aligned metadata CSV / JSON
- [ ] Build lightweight review frontend
- [ ] Add `human_truth.sqlite`
- [ ] Lock one baseline config:
  - one YOLO TRT model
  - one VLM model
  - one runtime mode
  - day + night clips
  - one VLM model
  - one runtime mode
- [ ] Run baseline daytime clip
- [ ] Run baseline nighttime clip
- [ ] Build config experiment matrix runner
- [ ] Compare only the most realistic VLM alternatives first
- [ ] Compare accuracy across selected VLM configs
- [ ] Compare speed across selected VLM configs
- [ ] Add YOLO / resolution sweeps if time remains
- [ ] Generate final report tables and summaries
