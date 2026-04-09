# ROI Layer

This layer restricts processing to the relevant scene region before detection.

## Public API

- `initialize_roi_layer`
- `update_roi_state`
- `apply_roi_to_frame`
- `build_roi_layer_package`

## Ownership

- Owns startup ROI discovery and the active ROI bounds.
- Passes through the full frame when ROI is disabled or not yet locked.
- Does not depend on tracking for ROI discovery.

## Quick Start

From the project root:

```powershell
python src/roi-layer/test_roi_layer.py
```
