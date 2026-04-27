# 2026-04-27 GRACE VLM Integration Plan

## Goal

Integrate the GRACE FHWA vehicle classifier into the existing pipeline as a VLM-layer backend, while preserving the current upstream and downstream layer contracts.

GRACE should live on the `vlm-layer` route instead of becoming a separate top-level pipeline branch. The current cropper -> VLM -> ack/package flow should remain the integration point.

All GRACE integration source should be moved under:

```text
src/vlm-layer/grace_integration/
```

The checkpoint remains local on the Jetson and ignored by git.

## Current Understanding

- GRACE is a 44M-parameter ConvNeXtV2 + Gaussian Transformer classifier, not a generative VLM.
- GRACE takes a 224x224 RGB vehicle crop.
- GRACE emits structured vehicle predictions:
  - FHWA class
  - vehicle type
  - axle count
  - trailer count
  - FHWA confidence
- The current VLM layer expects backend inference to return raw text that is parsed into the existing VLM package and ack package.
- Therefore, GRACE needs a translation layer that converts GRACE's structured prediction into the JSON-string shape the VLM layer already normalizes.
- The active semantic contract is moving from `is_truck` / `wheel_count` / `estimated_weight_kg` to `is_target_vehicle` / `axle_count` / GRACE classification fields.
- `estimated_weight_kg` is removed from the active VLM contract. It should not be emitted by GRACE or requested from prompt-based VLMs unless a future feature explicitly reintroduces weight estimation.

## Proposed Shape

Keep all integration work under `src/vlm-layer/`.

Preferred route:

1. Move the importable GRACE package under `src/vlm-layer/grace_integration/`.
2. Add a new VLM backend, `grace_fhwa`.
3. Add config support for `config_vlm_backend: grace_fhwa`.
4. Load GRACE through the VLM backend registry.
5. Run GRACE on the crop image received from `VLMFrameCropperLayerPackage`.
6. Convert GRACE output to a JSON string before returning from backend inference.
7. Let the existing `normalize_vlm_result`, `build_vlm_layer_package`, and ack flow continue to operate after updating them to the new `is_target_vehicle` / `axle_count` contract.
8. Add a YAML target-type file under `grace_integration` so GRACE target vehicle classes can be tuned without code changes.

## Translation Layer

GRACE output should be translated to a JSON string before it reaches the common VLM parser.

Example target shape:

```json
{
  "is_target_vehicle": true,
  "ack_status": "accepted",
  "retry_reasons": [],
  "confidence": 0.94,
  "grace_backend": "grace_fhwa",
  "fhwa_class": "FHWA-9",
  "fhwa_index": 7,
  "fhwa_confidence": 0.94,
  "vehicle_type": "semi_tractor",
  "vehicle_type_index": 14,
  "axle_count": 5.0,
  "trailer_count": "1"
}
```

The existing parser should be extended to preserve GRACE-specific fields in `vlm_layer_attributes`.

## Resolved Design Choices

- Use `axle_count` directly. Do not derive or depend on `wheel_count` in the active contract.
- Remove `estimated_weight_kg` from the active VLM contract.
- Use `is_target_vehicle`, not `is_truck`, across the cleaned-up VLM contract and docs.
- GRACE defaults to `ack_status: accepted` with no retry reasons when inference succeeds, because GRACE is a classifier and does not judge crop quality.
- Move GRACE source under `src/vlm-layer/grace_integration/`.
- Keep the GRACE checkpoint local on the Jetson, under the GRACE integration directory, and ignored by git.
- Store editable GRACE target vehicle type definitions in YAML under `src/vlm-layer/grace_integration/`.

## Local Artifact Policy

Commit source, docs, config, and adapter code.

Keep large local artifacts out of git:

- GRACE checkpoints local to the Jetson
- hydrated VLM model directories
- safetensors / pt / engine artifacts
- archived experimental model directories

## Implementation Steps

1. Add ignore rules for local-only model artifacts.
2. Move GRACE source into `src/vlm-layer/grace_integration/`.
3. Add `src/vlm-layer/grace_integration/target_vehicle_types.yaml`.
4. Add `grace_fhwa` to VLM backend/config allowed values.
5. Add `src/vlm-layer/backends/grace_fhwa.py`.
6. Implement GRACE model loading from the local Jetson package path.
7. Implement single and batch inference wrappers without adding new long-lived threads.
8. Implement `convert_grace_result_to_vlm_json`.
9. Update VLM normalization around `is_target_vehicle` and `axle_count`.
10. Extend VLM normalization so GRACE fields survive in `vlm_layer_attributes`.
11. Add unit tests for backend registry selection, YAML target type loading, and JSON translation.
12. Add a smoke path that runs GRACE on `truckimage.png` when local checkpoint files exist, skipping cleanly otherwise.
13. Update VLM and configuration docs.

## Validation

- Config validation accepts `config_vlm_backend: grace_fhwa`.
- Registry resolves GRACE backend explicitly and by model path where appropriate.
- Translation function produces parseable JSON.
- Translation output uses `is_target_vehicle` and `axle_count`, and does not include `estimated_weight_kg`.
- `normalize_vlm_result` preserves GRACE metadata.
- Existing SmolVLM/Qwen/Gemma backend tests still pass.
- GRACE smoke test skips when checkpoint is absent and runs when local checkpoint is present.
- Existing async/spill worker paths shut down cleanly; GRACE should reuse the VLM worker flow and should not introduce hanging background threads.

## Handoff Snapshot - 2026-04-27

Status at handoff: implementation is partially complete and intentionally left
uncommitted for review.

Already completed:

1. Moved the GRACE inference package from `src/GRACE_inference_package/` to
   `src/vlm-layer/grace_integration/`.
2. Updated `.gitignore` so the GRACE checkpoint remains local under
   `src/vlm-layer/grace_integration/checkpoint/`.
3. Added editable GRACE target vehicle type YAML:
   `src/vlm-layer/grace_integration/target_vehicle_types.yaml`.
4. Added new VLM backend module:
   `src/vlm-layer/backends/grace_fhwa.py`.
5. Updated backend registry/config schema so `config_vlm_backend: grace_fhwa`
   is accepted and can be resolved explicitly or by GRACE model path.
6. Updated VLM layer routing so GRACE initializes through the VLM layer,
   supports single and batch inference, and returns JSON strings through the
   shared VLM normalization path.
7. Updated the active VLM semantic contract toward:
   - `is_target_vehicle`
   - `axle_count`
   - GRACE metadata fields such as `fhwa_class`, `vehicle_type`,
     `trailer_count`, and confidence fields
8. Removed active emission/requesting of `estimated_weight_kg` and
   `wheel_count` from the new GRACE path and the VLM prompt contract.
9. Updated vehicle-state merge behavior so `is_target_vehicle=false` produces
   terminal status `no`; accepted target semantics produce `done`.
10. Updated deployment/review export code to write `is_target_vehicle`.
11. Updated benchmark plumbing in progress:
    - standalone backend/device matrix includes `grace_fhwa`
    - experiment matrix includes `grace_fhwa`
    - repo-root `benchmark.py` now passes configured `config_vlm_backend`
      into `VLMConfig`
    - several visualizer/headless utility entry points were started toward
      passing `config_vlm_backend` explicitly
12. Added tests for:
    - config validation accepting `grace_fhwa`
    - registry GRACE resolution
    - target vehicle YAML loading
    - GRACE output translation to VLM JSON

Focused validation already run once and passed:

```bash
python -m unittest \
  src/configuration-layer/test/test_config_node.py \
  src/vlm-layer/test/test_grace_fhwa_backend.py \
  src/vehicle-state-layer/test/test_vehicle_state_layer.py
```

Syntax validation already run once and passed for the core touched modules:

```bash
python -m py_compile \
  src/vlm-layer/layer.py \
  src/vlm-layer/backends/grace_fhwa.py \
  src/vlm-layer/backends/registry.py \
  pipeline/run_deployment_review.py \
  pipeline/run_experiment_matrix.py \
  pipeline/compare_against_human_truth.py \
  src/vehicle-state-layer/vehicle_state_layer.py \
  src/vlm-layer/util/visualize_vlm.py \
  src/roi-layer/util/visualize_roi_vlm.py
```

Known unfinished items before commit:

1. Re-run full syntax validation after the latest benchmark/utility plumbing
   edits.
2. Re-run focused unit tests after the latest benchmark/utility plumbing edits.
3. Check every `VLMConfig(...)` call for explicit backend propagation where it
   reads from config or CLI.
4. Decide whether to keep legacy parser compatibility for `is_truck` and
   `wheel_count`; current code keeps compatibility but does not emit those
   fields in the active GRACE contract.
5. Confirm `src/vlm-layer/test/smoke_test.py` CLI changes after syntax check.
6. Run the GRACE standalone backend benchmark on Jetson when ready:

```bash
python src/vlm-layer/test/benchmark_vlm_backend_device_matrix.py \
  --backends grace_fhwa \
  --devices cuda \
  --measured-runs 1 \
  --warmup-runs 0
```

7. Run the full pipeline benchmark after setting config to GRACE:

```yaml
config_vlm_enabled: true
config_vlm_backend: grace_fhwa
config_vlm_model: src/vlm-layer/grace_integration
```

Then:

```bash
python benchmark.py
```

Current handoff caution:

- The GRACE source move currently appears in git status as deletions under
  `src/GRACE_inference_package/` plus new files under
  `src/vlm-layer/grace_integration/`. Git should recognize these as renames
  once staged.
- The checkpoint is local-only and should not be committed.
- Do not commit until the unfinished validation items above are complete.
