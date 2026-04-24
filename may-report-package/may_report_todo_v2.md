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
  - `gemma_e2b_local`
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

Current stage decision:

- for the May-report performance pass, compare `cpu` vs `cuda` only
- run that comparison for all three modular VLM options:
  - `smolvlm_256m`
  - `qwen_0_8b`
  - `gemma_e2b_local`
- do not block this phase on INT8 / low-bit quantized VLM variants
- treat quantization effects as a later, separate experiment after the modular cpu/cuda comparison path is stable

Current findings behind that decision:

- CUDA is available on the Jetson used for these tests
- the active modular backends can target `cpu` and `cuda`
- the repo does not currently contain one clean, consistent INT8 VLM path across all three modular models
- attempting the unstable full three-backend `cpu/cuda` matrix on this Jetson has also been associated with:
  - SSH disconnect
  - ping failing to reach the host for about `5` minutes
  - eventual host recovery after that window
- treat those combinations as host-risky until they are re-run in a controlled manual session
- the archived quantized artifacts are mixed and not apples-to-apples with the active modular setup:
  - SmolVLM archived `int8` folder does not currently load cleanly as a modular backend checkpoint
  - Qwen has archived low-bit variants, but they are `4-bit` / AWQ / GPTQ rather than a clean INT8 path
  - Gemma is currently wired through local `llama.cpp` GGUF execution rather than a comparable Hugging Face INT8 path

Expected performance interpretation:

- `cpu` vs `cuda` is still a useful and valid report comparison because it isolates device-placement benefit for the modular VLM layer
- a true VLM INT8 comparison would answer a different question: quantization benefit on top of device placement
- INT8 on CUDA could be faster than the current fp16 / bf16 paths, but we should not assume a large universal speedup without a working apples-to-apples checkpoint and loader
- on Jetson-class hardware, actual gain depends on whether the run is compute-bound or memory / runtime-overhead-bound; practical improvement can range from modest to substantial, but cannot be defended in the report until we have a stable INT8 path for the same model family
- for the current report stage, document that any INT8 speedup is unmeasured and out of scope rather than estimated as a hard percentage
- once the backend/device comparison is stable, revisit quantization as a follow-on experiment with explicit runtime support and matched model families

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

- [x] Add `review-package/` top-level folder convention
- [x] Modularize VLM backend selection
- [x] Build headless review artifact generator
- [x] Save new-track images
- [x] Save accepted target crops
- [x] Save aligned metadata CSV / JSON
- [x] Build lightweight review frontend
- [x] Add `human_truth.sqlite`
- [ ] Lock one baseline config:
  - one YOLO TRT model
  - one VLM model
  - one runtime mode
  - day + night clips
  - one VLM model
  - one runtime mode
- [x] Run baseline daytime clip
- [ ] Run baseline nighttime clip
- [x] Build config experiment matrix runner
- [ ] Compare only the most realistic VLM alternatives first
- [ ] Compare accuracy across selected VLM configs
- [ ] Compare speed across selected VLM configs
- [ ] Add YOLO / resolution sweeps if time remains
- [ ] Generate final report tables and summaries

Implementation note:

- `pipeline/run_experiment_matrix.py` now exists and defaults to safe mode:
  - stable cases can be run normally
  - currently risky cases are skipped unless `--allow-unstable` is passed on purpose
- `pipeline/build_report_summary.py` now exists to turn a matrix JSON artifact into report-ready `json` / `md` / `csv`
- baseline-lock work is now partially implemented:
  - dedicated config file: `src/configuration-layer/config.report-baseline.yaml`
  - dedicated runner support: `pipeline/run_deployment_review.py --config <yaml>`
  - dedicated pair-planning support: `pipeline/run_report_baseline_pair.py`
  - current locked baseline candidate on Jetson:
    - YOLO TRT on `cuda`
    - `smolvlm_256m`
    - VLM on `cpu`
    - VLM runtime mode `async`
  - current planned baseline clip pair:
    - day: `data/upson1.mp4`
    - night: `data/sample1.mp4`
  - first `cuda/cuda` baseline attempt was rejected because it crashed at first VLM dispatch with `CUBLAS_STATUS_ALLOC_FAILED`
  - patched `cuda/cpu` baseline daytime validation is currently in progress on `data/upson1.mp4`
  - that active baseline process still requires explicit completion verification:
    - confirm `summaries/run_summary.json` exists
    - confirm a terminal `run_summary` event was written
    - confirm the process exited cleanly
  - the baseline pair helper has already been dry-run validated and writes plan artifacts under `review-package/artifacts/`
- comparison-planning work is now partially implemented for the next TODO:
  - `pipeline/run_experiment_matrix.py --preset may_report_realistic --dry-run`
    - resolves the current recommended first-pass cases:
      - `smolvlm_256m` on `cpu`
      - `smolvlm_256m` on `cuda`
      - `gemma_e2b_local` on `cpu`
    - excludes the currently non-realistic / host-risky paths from the first-pass plan
  - this selection workflow was validated end-to-end through `pipeline/build_report_summary.py`
  - `pipeline/build_report_summary.py` now also treats dry-run plans as first-class outputs:
    - counts `planned_cases`
    - carries `selection_reason` into report artifacts
  - a future measured run is still required before the TODO item itself can be marked complete
- selected-config speed comparison is now partially implemented:
  - `pipeline/build_selected_vlm_comparison.py`
    - combines the realistic first-pass plan with the current measured smol results and Gemma follow-up result
    - writes repo-visible `json` / `md` / `csv` outputs
  - current artifact recommendation:
    - use `smolvlm_256m` on `cuda` as the first measured performance reference on this Jetson
  - this still does not replace the need for future measured runs of any not-yet-measured selected cases
- selected-config accuracy comparison is now partially implemented:
  - `pipeline/build_selected_vlm_accuracy_comparison.py`
    - builds repo-visible `json` / `md` / `csv` tables from current `compare_against_human_truth` artifacts
  - current output is intentionally sparse because only one reviewed truth-comparison artifact exists so far
  - this gives the report workflow a stable table format now, while leaving room for more reviewed runs later
- consolidated final-report-table work is now partially implemented:
  - `pipeline/build_may_report_tables.py`
    - aggregates current baseline-pair planning artifacts
    - aggregates current matrix artifacts
    - aggregates current truth-comparison artifacts
    - scans `review-package/runs/` for baseline run completion state
  - validated output artifacts now exist under `review-package/artifacts/`
  - this validation also confirmed:
    - patched daytime baseline run `20260424_001056_upson1_may_baseline_day` completed successfully
  - navigation/indexing support now also exists:
    - `pipeline/build_may_report_artifact_index.py`
    - latest report outputs can be discovered through one generated index file instead of hunting across filenames
  - nighttime baseline run is now in progress on `data/sample1.mp4` using the same locked config and still needs explicit completion verification
