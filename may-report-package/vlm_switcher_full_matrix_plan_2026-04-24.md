# VLM Switcher Full Matrix Plan

Date: `2026-04-24`

## Goal

Answer the actual open question directly:

- do all `3` intended VLM backends run through the switcher on both `cpu` and `cuda`
- if not, which cases:
  - pass
  - fail cleanly
  - are blocked by host-risk instability

## Backends In Scope

- `smolvlm_256m`
- `qwen_0_8b`
- `gemma_e2b_local`

## Devices In Scope

- `cpu`
- `cuda`

## Validation Rule

A case is only considered `passed` if it reaches real model initialization and real inference through the switcher path.

Safe config-resolution alone does not count as a matrix pass.

## Execution Order

1. [x] `smolvlm_256m` on `cpu`
2. [x] `smolvlm_256m` on `cuda`
3. [x] `gemma_e2b_local` on `cpu`
4. [x] `qwen_0_8b` on `cpu`
5. [x] `gemma_e2b_local` on `cuda`
6. [x] `qwen_0_8b` on `cuda`

## Status Labels

- `passed`
  - real switcher init + real inference completed
- `failed`
  - case ran and returned a concrete runtime error
- `blocked_host_risk`
  - case is not being treated as safely runnable on the Jetson because prior attempts were associated with host instability
- `not_run_yet`
  - not executed yet in this pass

## Current Prior Evidence

- `smolvlm_256m` on `cpu`
  - previously passed in bounded switcher validation
- `smolvlm_256m` on `cuda`
  - previously passed in bounded switcher validation
- `gemma_e2b_local` on `cpu`
  - prior benchmark attempt failed with:
    - `Server disconnected without sending a response`
- `gemma_e2b_local` on `cuda`
  - prior work marked this as unstable / non-viable on this Jetson
- `qwen_0_8b` on `cpu`
  - prior work marked this as unstable on this Jetson
- `qwen_0_8b` on `cuda`
  - prior work marked this as unstable on this Jetson

## Result Table

| Backend | Device | Status | Evidence | Notes |
| --- | --- | --- | --- | --- |
| `smolvlm_256m` | `cpu` | `passed` | `src/vlm-layer/test/run_config_vlm_once.py --config src/configuration-layer/config.report-baseline.yaml --target-ids 1` | bounded live switcher validation completed with real inference |
| `smolvlm_256m` | `cuda` | `passed` | `src/vlm-layer/test/run_config_vlm_once.py --config src/configuration-layer/config.jetson.yaml --target-ids 1` | bounded live switcher validation completed with real inference |
| `gemma_e2b_local` | `cpu` | `failed` | `src/configuration-layer/config.vlm-switcher-gemma-cpu.yaml` + `review-package/artifacts/vlm_switcher_gemma_cpu_20260424_rerun` | config-driven CPU selection is now confirmed, but real inference still fails with `RemoteProtocolError: Server disconnected without sending a response.` |
| `qwen_0_8b` | `cpu` | `passed` | `src/configuration-layer/config.vlm-switcher-qwen-hydrated-cpu.yaml` + `review-package/artifacts/vlm_switcher_qwen_hydrated_cpu_20260424` | passed after building a local hydrated checkpoint directory that combines repo metadata with real cached tokenizer/weight files; original repo-local Qwen path is still broken by unhydrated LFS pointers |
| `gemma_e2b_local` | `cuda` | `failed` | `src/configuration-layer/config.vlm-switcher-gemma-cuda.yaml` + `review-package/artifacts/vlm_switcher_gemma_cuda_20260424` | config-driven CUDA selection is confirmed, but startup fails with GPU OOM during model load |
| `qwen_0_8b` | `cuda` | `failed` | `src/configuration-layer/config.vlm-switcher-qwen-hydrated-cuda.yaml` + `review-package/artifacts/vlm_switcher_qwen_hydrated_cuda_20260424_recheck_final` | hydrated checkpoint is locally runnable, but the Jetson CUDA path still fails in bounded validation with a repo-classified CUDA runtime capacity/stability error during model load/inference rather than a switcher-resolution bug |

## Exit Condition

This matrix is only “fully working” if every row above is marked `passed`.

If one or more rows end in `failed` or `blocked_host_risk`, then the honest conclusion is that the switcher is only partially working on this Jetson for the tested matrix.

## Current Recheck Notes

Later current-branch revalidation refined the non-Smol cases:

- `qwen_0_8b`
  - the repo-local path is still broken by Git LFS pointer files
  - the hydrated cache snapshot by itself is also not directly runnable through the current setup because the snapshot directory lacks a usable `config.json`
  - a local repaired directory now exists:
    - `src/vlm-layer/Qwen3.5-0.8B-local-hydrated`
  - bounded CPU validation now passes through:
    - `src/configuration-layer/config.vlm-switcher-qwen-hydrated-cpu.yaml`
    - `review-package/artifacts/vlm_switcher_qwen_hydrated_cpu_20260424/`
  - bounded CUDA validation now gets past packaging and model load, but fails during CUDA compute:
    - `src/configuration-layer/config.vlm-switcher-qwen-hydrated-cuda.yaml`
    - `review-package/artifacts/vlm_switcher_qwen_hydrated_cuda_20260424/`
    - `review-package/artifacts/vlm_switcher_qwen_hydrated_cuda_20260424_recheck_final/`
  - latest current-branch recheck after backend error-handling cleanup:
    - hydrated CPU path still passes:
      - `review-package/artifacts/vlm_switcher_qwen_hydrated_cpu_20260424_recheck/`
    - hydrated CUDA path still fails, but the backend now reports it more honestly as:
      - `CUDA inference failed for this Hugging Face VLM on Jetson during load_model`
      - interpretation:
        - real CUDA runtime capacity/stability failure on this Jetson
        - not a config-resolution bug
  - result:
    - `cpu` is now repairable and has a passing bounded config-driven validation path
    - `cuda` is now narrowed to a runtime/device execution failure rather than a packaging failure
- `gemma_e2b_local`
  - a helper bug was fixed so `config_vlm_device: cpu` is now actually honored in the bounded one-shot validator
  - result after the fix:
    - `cpu` really switches to CPU, but inference still fails with `RemoteProtocolError`
    - `cuda` really switches to CUDA, but startup fails with GPU OOM
  - later rerun to answer the “it should fit in 8 GB” question:
    - the CUDA path still fails during `llama-server` model load
    - server log shows:
      - model type `E2B`
      - about `4.65 B` params
      - failure when allocating another about `436.99 MiB` CUDA buffer
    - practical reading:
      - GGUF file size on disk is not the full live CUDA working set
      - Jetson `8 GB` is shared unified memory, not empty dedicated VRAM for this model alone
      - YOLO/TensorRT, `mmproj`, llama.cpp scratch buffers, and general process overhead consume part of that budget before Gemma finishes loading
    - repo conclusion:
      - treat `gemma_e2b_local` on CUDA as a current Jetson capacity/runtime limit, not a switcher-resolution problem

Current honest reading:

- `smolvlm_256m` is the only backend currently shown to switch successfully across both `cpu` and `cuda` in bounded config-driven validation on this Jetson
- `qwen_0_8b` now has a passing bounded CPU path when the checkpoint is hydrated locally, but the repo-local packaged path is still broken and the CUDA path currently fails with a reproducible Jetson CUDA runtime capacity/stability error in bounded validation
- `gemma_e2b_local` honors switching now, but neither device path currently completes a successful bounded inference run
