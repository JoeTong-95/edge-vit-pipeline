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
- Deduplicates overlapping startup detections locally, so the ROI lock threshold behaves like a unique-ish vehicle count rather than a raw per-frame detection count.

## Quick Start

From the project root:

```powershell
python src/roi-layer/test_roi_layer.py
python src/roi-layer/visualize_roi.py --show
```

Other config-driven visualizers that read `src/configuration-layer/config.yaml` are listed in `pipeline/README.md` (including YOLO, tracking, cropper, and end-to-end VLM).
