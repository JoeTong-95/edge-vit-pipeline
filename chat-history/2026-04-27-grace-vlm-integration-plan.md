# 2026-04-27 GRACE VLM Integration Plan

## Goal

Integrate the GRACE FHWA vehicle classifier into the existing pipeline as a VLM-layer backend, while preserving the current upstream and downstream layer contracts.

GRACE should live on the `vlm-layer` route instead of becoming a separate top-level pipeline branch. The current cropper -> VLM -> ack/package flow should remain the integration point.

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
- Therefore, GRACE needs a translation layer that converts GRACE's structured prediction into the same JSON-string shape the VLM layer already normalizes.

## Proposed Shape

Keep all integration work under `src/vlm-layer/`.

Preferred route:

1. Move or mirror the importable GRACE package under the VLM-layer model/backend path.
2. Add a new VLM backend, likely `grace_fhwa`.
3. Add config support for `config_vlm_backend: grace_fhwa`.
4. Load GRACE through the VLM backend registry.
5. Run GRACE on the crop image received from `VLMFrameCropperLayerPackage`.
6. Convert GRACE output to a JSON string before returning from backend inference.
7. Let the existing `normalize_vlm_result`, `build_vlm_layer_package`, and ack flow continue to operate.

## Translation Layer

GRACE output should be translated to a JSON string before it reaches the common VLM parser.

Example target shape:

```json
{
  "is_truck": true,
  "wheel_count": 10,
  "estimated_weight_kg": 0,
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

## Open Design Choices

- Whether `wheel_count` should be derived as `round(axle_count * 2)` or replaced with a clearer downstream field such as `axle_count`.
- Whether `estimated_weight_kg` should remain `0` for GRACE or be estimated from FHWA class later.
- Whether GRACE should always return `ack_status: accepted`, since it does not naturally produce crop-quality retry reasons.
- Whether `GRACE_inference_package` should be physically moved to `src/vlm-layer/GRACE_inference_package/` or wrapped from its current location first and moved after validation.

## Local Artifact Policy

Commit source, docs, config, and adapter code.

Keep large local artifacts out of git:

- GRACE checkpoints
- hydrated VLM model directories
- safetensors / pt / engine artifacts
- archived experimental model directories

## Implementation Steps

1. Add ignore rules for local-only model artifacts.
2. Add `grace_fhwa` to VLM backend/config allowed values.
3. Add `src/vlm-layer/backends/grace_fhwa.py`.
4. Implement GRACE model loading from a local package path.
5. Implement single and batch inference wrappers.
6. Implement `convert_grace_result_to_vlm_json`.
7. Extend VLM normalization so GRACE fields survive in `vlm_layer_attributes`.
8. Add unit tests for backend registry selection and JSON translation.
9. Add a smoke path that runs GRACE on `truckimage.png` when local checkpoint files exist, skipping cleanly otherwise.
10. Update VLM and configuration docs.

## Validation

- Config validation accepts `config_vlm_backend: grace_fhwa`.
- Registry resolves GRACE backend explicitly and by model path where appropriate.
- Translation function produces parseable JSON.
- `normalize_vlm_result` preserves GRACE metadata.
- Existing SmolVLM/Qwen/Gemma backend tests still pass.
- GRACE smoke test skips when checkpoint is absent and runs when local checkpoint is present.
