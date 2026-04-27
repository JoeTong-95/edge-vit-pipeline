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
- `done`: VLM accepted the crop as a target vehicle and semantic JSON was stored
- `no`: VLM rejected the track as not a target vehicle for the active backend
- `dead`: the track aged out on the cropper-side lost-threshold path before a full semantic round completed

## Quick Start

From the project root:

```powershell
python src/vehicle-state-layer/test/test_vehicle_state_layer.py
```

The orchestration script `src/vlm-layer/util/visualize_vlm.py` calls `update_vehicle_state_from_tracking`, `update_vehicle_state_from_vlm_ack`, and `update_vehicle_state_from_vlm` (on accept/finalize) to exercise the documented ack loop next to the cropper layer.

## Current Session Note

The active VLM prompt in this branch no longer asks for `truck_type`.

`vehicle_state_layer_truck_type` therefore acts as a legacy compatibility field
for older downstream expectations. The active VLM semantic payload now centers
on:

- `is_target_vehicle`
- `axle_count`
- `vlm_ack_status`
- `vlm_retry_reasons`

`estimated_weight_kg` and `wheel_count` are no longer part of the active VLM
contract.

## Changes

Layer changes in this branch

- Updated compatibility with the simplified active VLM contract used in this
  session: the current prompt no longer asks for `truck_type`, so
  `vehicle_state_layer_truck_type` now behaves as a legacy compatibility slot
  rather than an actively populated semantic target.
- Updated `README.md` to explain that the active VLM semantic payload now
  centers on `is_target_vehicle`, `axle_count`,
  `vlm_ack_status`, and `vlm_retry_reasons`.

- Added `vehicle_state_layer.py` implementing the pipeline contract public API:
  `initialize_vehicle_state_layer`, `update_vehicle_state_from_tracking`,
  `update_vehicle_state_from_vlm`, `get_vehicle_state_record`, and
  `build_vehicle_state_layer_package`.
- Kept ownership aligned with the pipeline doc: this layer stores persistent
  metadata and does not re-decide `new` vs `active` vs `lost`.
- Tightened VLM enrichment so semantic results only apply to records already
  created by tracking, matching the intended optional enrichment path.
- Added integration compatibility so the layer accepts the real `VLMLayerPackage`
  dataclass produced by the VLM layer, not only dict-style fixtures.
- Added `test_vehicle_state_layer.py` coverage for tracking updates, VLM merges,
  stale-record pruning, and VLM dataclass compatibility.
- Added VLM acknowledgement tracking fields and `update_vehicle_state_from_vlm_ack` so retry/finalize decisions are persisted per track.

## 2026-04-09

- Updated `README.md` to reference `visualize_vlm.py` as an integration exercise for tracking + VLM ack + enrichment updates.

## 2026-04-10

- Added `vehicle_state_layer_terminal_status` so persistent per-track state now
  records whether a vehicle is still `tracking`, has reached semantic `done`,
  or is terminal `dead`.
- Updated VLM merge and ack merge behavior so `is_target_vehicle=false` marks a
  track `no`, while accepted target-vehicle semantics mark the track `done`.
- Split terminal rejection semantics so VLM label rejection now marks a track
  `no`, while `dead` remains reserved for the lost-threshold path.
