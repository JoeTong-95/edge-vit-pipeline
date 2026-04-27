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
- The active semantic contract is changing from `is_truck` / `wheel_count` / `estimated_weight_kg` to `is_target_vehicle` / `axle_count` / GRACE classification fields.
- `estimated_weight_kg` is removed from the active VLM contract. It should not be emitted by GRACE or requested from prompt-based VLMs unless a future feature explicitly reintroduces weight estimation.

## Proposed Shape

Keep all integration work under `src/vlm-layer/`.

Preferred route:

1. Move the importable GRACE package under `src/vlm-layer/grace_integration/`.
2. Add a new VLM backend, likely `grace_fhwa`.
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
