# VLM Switcher Validation Plan

Date: `2026-04-24`

## Goal

Validate the VLM switcher end-to-end in increasing risk order so we can trust config-specific review-package generation.

## Guardrails

- Previous sessions that smoke-ran unstable backend/device combinations on this Jetson were associated with SSH disconnect / temporary host disappearance.
- Work should progress from no-inference validation to bounded live inference checks.
- Each step must be documented before moving to the next step.
- If a step destabilizes or hangs, stop escalation and record the failure mode instead of broadening the test matrix.

## Step Plan

1. [x] Safe config-resolution validation
   - Validate all in-repo YAML configs without launching inference.
   - Confirm:
     - effective VLM backend
     - effective VLM device
     - effective runtime mode
     - local checkpoint readiness
   - Completed in this cycle with `pipeline/validate_vlm_switcher.py`.
   - Observed results:
     - `config.report-baseline.yaml` -> `smolvlm_256m` / `cpu` / `async`
     - `config.yaml` -> `smolvlm_256m` / `cuda` / `async`
     - `config.jetson.yaml` -> `smolvlm_256m` / `cuda` / `async`
     - `config.cpu-test.yaml` -> VLM disabled as expected

2. [x] Bounded live validation on the lowest-risk path
   - Use the locked report baseline config:
     - backend `smolvlm_256m`
     - VLM device `cpu`
     - runtime `async`
   - Run a bounded one-shot validation that exercises:
     - config load
     - YOLO init
     - tracking
     - cropper
     - VLM initialization
     - at least one real VLM inference result
   - Completed with:
     - `python3 src/vlm-layer/test/run_config_vlm_once.py --config src/configuration-layer/config.report-baseline.yaml --max-frames 60 --target-ids 1 --output-dir review-package/artifacts/vlm_switcher_validation_step2_baseline_cpu`
   - Observed result:
     - YOLO initialized on `cuda`
     - VLM initialized from the baseline config on `cpu`
     - first real dispatch completed at frame `4`
     - saved debug image:
       - `review-package/artifacts/vlm_switcher_validation_step2_baseline_cpu/frame_00004__track_1__initial_candidate.png`
     - run exited cleanly after capturing `1 / 1` target IDs

3. [x] Bounded live validation on the next higher-risk path
   - Use the default Jetson-style CUDA VLM path:
     - backend `smolvlm_256m`
     - VLM device `cuda`
     - runtime `async`
   - Keep the run tightly bounded.
   - Treat this as the first meaningful live switcher stress check after the CPU baseline.
   - Completed with:
     - `python3 src/vlm-layer/test/run_config_vlm_once.py --config src/configuration-layer/config.jetson.yaml --max-frames 60 --target-ids 1 --output-dir review-package/artifacts/vlm_switcher_validation_step3_cuda`
   - Observed result:
     - YOLO initialized on `cuda`
     - VLM initialized from the config on `cuda`
     - first real dispatch completed at frame `4`
     - saved debug image:
       - `review-package/artifacts/vlm_switcher_validation_step3_cuda/frame_00004__track_1__initial_candidate.png`
     - run exited cleanly after capturing `1 / 1` target IDs

4. [x] Compare bounded live results against expected config resolution
   - Confirm the live path matches the config-driven backend/device/runtime chosen in step 1.
   - Record whether failures are:
     - config-resolution bugs
     - initialization bugs
     - runtime stability bugs
   - Observed match:
     - baseline config resolved to `smolvlm_256m` / `cpu` / `async` and the bounded live run initialized VLM on `cpu`
     - Jetson config resolved to `smolvlm_256m` / `cuda` / `async` and the bounded live run initialized VLM on `cuda`
   - Current interpretation:
     - no config-resolution bug was observed in the tested paths
     - no initialization failure was observed in the bounded tested paths
     - no immediate runtime stability failure was observed in the bounded tested paths

5. [x] Mark switcher validation status for package generation
   - If step 2 and step 3 both pass:
     - mark the tested config paths as e2e validated for review-package generation
   - If only step 2 passes:
     - mark only the baseline CPU path as e2e validated
     - leave CUDA path as unresolved runtime risk
   - Result for this cycle:
     - the tested `smolvlm_256m` CPU baseline path is e2e validated
     - the tested `smolvlm_256m` CUDA Jetson path is also e2e validated in a bounded one-shot run
     - review-package generation can proceed for these tested config paths
   - Remaining unvalidated area:
     - other backend families such as `qwen_0_8b` and `gemma_e2b_local`
     - longer-duration runtime stability beyond the bounded one-shot validation used here

## Status Log

- `pending`: step not yet run
- `in_progress`: currently executing
- `complete`: step finished and documented
- `blocked`: step hit a concrete failure and should not be escalated without a new decision

## Current Branch Recheck

Later in the same day, the `smolvlm_256m` `cpu` vs `cuda` switcher path was revalidated again on the current branch state so the result did not depend only on the earlier one-shot notes.

Recheck evidence:

- safe config-resolution still matched expectations:
  - `config.report-baseline.yaml` -> `smolvlm_256m` / `cpu` / `async`
  - `config.jetson.yaml` -> `smolvlm_256m` / `cuda` / `async`
- bounded one-shot live recheck still passed for both paths:
  - `review-package/artifacts/vlm_switcher_validation_20260424_cpu_recheck/`
  - `review-package/artifacts/vlm_switcher_validation_20260424_cuda_recheck/`
- a slightly larger bounded helper workload also passed for both paths:
  - `review-package/artifacts/vlm_switcher_validation_20260424_cpu_two_targets/`
  - `review-package/artifacts/vlm_switcher_validation_20260424_cuda_two_targets/`

Important interpretation:

- this recheck increases confidence that the config-driven `smolvlm_256m` switcher works for bounded helper workloads on both `cpu` and `cuda`
- it does **not** change the earlier report-stage finding that the full deployment-review path with `YOLO cuda + VLM cuda` was not a stable May-report baseline on this Jetson
- for report-package generation, the locked stable baseline remains:
  - YOLO on `cuda`
  - `smolvlm_256m` on `cpu`
  - runtime `async`
