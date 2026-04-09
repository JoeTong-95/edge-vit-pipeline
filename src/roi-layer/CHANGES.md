Layer changes in this branch

- Added `roi_layer.py` implementing the pipeline contract public API:
  `initialize_roi_layer`, `update_roi_state`, `apply_roi_to_frame`, and
  `build_roi_layer_package`.
- Implemented the documented ROI node helpers for candidate-box collection,
  ROI-bound computation, bound locking, cropping, and passthrough behavior.
- Added integration compatibility so ROI accepts both dict-style input fixtures
  and the real `InputLayerPackage` dataclass emitted by the input layer on the
  confirmed main path.
- Added `test_roi_layer.py` covering disabled passthrough, startup discovery,
  ROI lock, cropped output, and dataclass-package compatibility.
- Added `visualize_roi.py` to render ROI discovery, ROI lock state, and a live
  side-by-side view of the full frame and active ROI crop.
