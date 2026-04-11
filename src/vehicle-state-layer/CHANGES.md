Layer changes in this branch

- Updated compatibility with the simplified active VLM contract used in this
  session: the current prompt no longer asks for `truck_type`, so
  `vehicle_state_layer_truck_type` now behaves as a legacy compatibility slot
  rather than an actively populated semantic target.
- Updated `README.md` to explain that the active VLM semantic payload now
  centers on `is_truck`, `wheel_count`, `estimated_weight_kg`,
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
- Updated VLM merge and ack merge behavior so `is_truck=false` marks a track
  `dead`, while accepted truck semantics mark the track `done`.
- Split terminal rejection semantics so VLM label rejection now marks a track
  `no`, while `dead` remains reserved for the lost-threshold path.

