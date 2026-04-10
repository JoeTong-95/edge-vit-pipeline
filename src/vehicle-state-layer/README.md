# Vehicle State Layer

This layer stores persistent per-vehicle metadata across frames.

## Public API

- `initialize_vehicle_state_layer`
- `update_vehicle_state_from_tracking`
- `update_vehicle_state_from_vlm`
- `update_vehicle_state_from_vlm_ack`
- `get_vehicle_state_record`
- `build_vehicle_state_layer_package`

## Ownership

- Owns persistent metadata such as first seen / last seen, lost-frame count,
  stored vehicle class, semantic tags, truck subtype, and whether VLM has been
  called.
- Stores VLM acknowledgement state for retry/finalize orchestration.
- Does not own the `new` vs `active` vs `lost` decision.

## VLM Lifecycle Fields

The package now also exposes:

- `vehicle_state_layer_vlm_ack_status`
- `vehicle_state_layer_vlm_retry_requested`
- `vehicle_state_layer_vlm_final_candidate_sent`

## Quick Start

From the project root:

```powershell
python src/vehicle-state-layer/test_vehicle_state_layer.py
```

The orchestration script `src/vlm-layer/visualize_vlm.py` calls `update_vehicle_state_from_tracking`, `update_vehicle_state_from_vlm_ack`, and `update_vehicle_state_from_vlm` (on accept/finalize) to exercise the documented ack loop next to the cropper layer.