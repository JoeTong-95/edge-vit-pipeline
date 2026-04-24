# SmolVLM 256M Optimization Plan

Date: `2026-04-24`

## Why This Exists

The switcher work is now far enough along that the next useful pass should focus on optimizing the one backend that is already shown to work on this Jetson:

- `smolvlm_256m`

Do **not** spend the next pass reopening:

- `gemma_e2b_local` CUDA fit questions
- repo-local `qwen_0_8b` packaging repair

Those issues are already documented elsewhere and are not the best report-stage optimization target.

## Current Stable Reading

- locked May-report baseline:
  - YOLO `cuda`
  - `smolvlm_256m` VLM `cpu`
  - runtime `async`
- bounded helper-path validation also passed for:
  - `smolvlm_256m` on `cuda`
- current CUDA comparison configs now exist for the same Smol backend:
  - `src/configuration-layer/config.vlm-switcher-smol-cuda-og.yaml`
  - `src/configuration-layer/config.vlm-switcher-smol-cuda-optimized.yaml`

## Current Branch Finding

The first bounded optimization pass has already refined what “optimized” should mean on this Jetson.

- original comparison config:
  - `config.vlm-switcher-smol-cuda-og.yaml`
  - batch size `1`
  - batch wait `24 ms`
- tested optimized variants:
  - batch size `4` + wait `50 ms`
    - reduced wall time
    - but produced worker-error rows from CUDA batch-capacity failures
    - not selected as the keeper
  - batch size `2` + wait `50 ms`
    - completed cleanly
    - kept the same bounded result count as the OG run
    - reduced bounded run duration on `data/sample1.mp4` / `60` frames from about `24.192 s` to about `16.491 s`
- current preferred tuned comparison config:
  - `config.vlm-switcher-smol-cuda-optimized.yaml`
  - batch size `2`
  - batch wait `50 ms`

## Goal

Use the VLM switcher to compare a controlled “before vs after” Smol CUDA path, then keep the changes that improve throughput or stability without undermining the locked report baseline.

## Scope For The Next Agent

Work only on `smolvlm_256m` optimization.

Start from:

1. `config.vlm-switcher-smol-cuda-og.yaml`
2. `config.vlm-switcher-smol-cuda-optimized.yaml`

Treat these as the first comparison pair.

## First Optimization Direction

Prefer stack/runtime efficiency before model-format or engine-swap work.

First levers to tune:

- `config_vlm_worker_batch_size`
- `config_vlm_worker_batch_wait_ms`
- async queue behavior
- spill/deferred-queue behavior for backlog-heavy clips

## Guardrails

- Keep all runs bounded and artifact-backed.
- Do not change the locked May-report baseline config just to chase speed.
- Do not broaden the optimization pass into Qwen or Gemma rescue work.
- If a CUDA optimization attempt regresses stability, record it and fall back rather than forcing it into the baseline path.

## Suggested Execution Order

1. Revalidate `cuda-og` and `cuda-optimized` through the switcher helper.
2. Measure whether the tuned async config helps materially on a bounded real run.
3. Keep `batch=2` as the current tuned setting unless a later pass finds a better stable point.
4. Compare results with the existing helper/matrix tooling.
5. Keep the report baseline on CPU unless a stronger full-path stability result is shown.

## Success Condition

A good outcome is not necessarily “move baseline to CUDA.”

A good outcome is:

- one clear, reproducible Smol optimization comparison
- one or more repo-visible configs for that comparison
- honest notes about whether the optimized CUDA path is:
  - faster
  - equally stable
  - or only suitable for bounded helper/perf experiments
