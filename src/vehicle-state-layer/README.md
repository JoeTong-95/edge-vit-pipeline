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
  stored vehicle class, semantic tags, legacy truck-type slot, and whether VLM has been
  called.
- Stores VLM acknowledgement state for retry/finalize orchestration.
- Does not own the `new` vs `active` vs `lost` decision.

## VLM Lifecycle Fields

The package now also exposes:

- `vehicle_state_layer_vlm_ack_status`
- `vehicle_state_layer_vlm_retry_requested`
- `vehicle_state_layer_vlm_final_candidate_sent`
- `vehicle_state_layer_terminal_status`

`vehicle_state_layer_terminal_status` is the persistent per-track end-state used by
the current VLM path:

- `tracking`: normal tracked flow, no terminal VLM decision yet
- `done`: VLM accepted the crop as a truck and semantic JSON was stored
- `no`: VLM rejected the track as not one of the currently flagged detector labels
- `dead`: the track aged out on the cropper-side lost-threshold path before a full semantic round completed

## Quick Start

From the project root:

```powershell
python src/vehicle-state-layer/test_vehicle_state_layer.py
```

The orchestration script `src/vlm-layer/visualize_vlm.py` calls `update_vehicle_state_from_tracking`, `update_vehicle_state_from_vlm_ack`, and `update_vehicle_state_from_vlm` (on accept/finalize) to exercise the documented ack loop next to the cropper layer.

## Current Session Note

The active VLM prompt in this branch no longer asks for `truck_type`.

`vehicle_state_layer_truck_type` therefore acts as a compatibility field for
older downstream expectations, while the active VLM semantic payload now mainly
centers on:

- `is_truck`
- `wheel_count`
- `estimated_weight_kg`
- `vlm_ack_status`
- `vlm_retry_reasons`
