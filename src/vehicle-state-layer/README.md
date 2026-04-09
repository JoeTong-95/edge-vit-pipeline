# Vehicle State Layer

This layer stores persistent per-vehicle metadata across frames.

## Public API

- `initialize_vehicle_state_layer`
- `update_vehicle_state_from_tracking`
- `update_vehicle_state_from_vlm`
- `get_vehicle_state_record`
- `build_vehicle_state_layer_package`

## Ownership

- Owns persistent metadata such as first seen / last seen, lost-frame count,
  stored vehicle class, semantic tags, truck subtype, and whether VLM has been
  called.
- Does not own the `new` vs `active` vs `lost` decision. That remains in the
  tracking layer.

## Quick Start

From the project root:

```powershell
python src/vehicle-state-layer/test_vehicle_state_layer.py
```
