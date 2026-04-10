Layer changes in this branch

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

