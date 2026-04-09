Layer changes in this branch

- Added `vlm_frame_cropper_layer.py` implementing the pipeline contract public
  API: `build_vlm_frame_cropper_request_package`, `extract_vlm_object_crop`, and
  `build_vlm_frame_cropper_package`.
- Implemented the documented node helpers for source-frame resolution, bbox crop
  extraction, and crop validation.
- Added integration compatibility so the cropper accepts the real
  `InputLayerPackage` dataclass emitted by the input layer, not only dict-style
  test fixtures.
- Kept the package field names exactly aligned with the pipeline contract for
  both request and VLM-ready crop packages.
- Added `test_vlm_frame_cropper_layer.py` coverage for disabled-mode request
  suppression, dataclass input compatibility, crop extraction, and package output.
- Added `visualize_vlm_frame_cropper.py` to render tracked-vehicle overlays and
  a crop contact sheet showing each vehicle image that gets passed through to the
  VLM frame cropper layer.
