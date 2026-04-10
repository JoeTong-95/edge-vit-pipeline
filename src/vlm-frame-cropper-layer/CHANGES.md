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
- Added internal per-track crop-cache helpers so candidate crops can be stored
  locally, survive temporary `lost` tracker states, and keep a best current VLM
  candidate without changing the documented public cropper API.
- Updated the cropper visualizer to render one row per tracked vehicle, fill the
  cache sequence according to `config_vlm_crop_cache_size`, and show the current
  selected VLM candidate in a dedicated final column.
- Updated `visualize_vlm_frame_cropper.py` so it only writes a video when `--output` is provided, instead of auto-saving into `data/` on every run.
- Exposed named crop-selection weight and normalizer constants in `vlm_frame_cropper_layer.py`, and documented where to tune them in the layer README and visualizer.
- Implemented the one-shot VLM dispatch loop: collect until cache full, send one best crop, reopen only on explicit retry ack, and finalize with the best available crop when the track leaves scope.
- Trimmed per-track crop cache by top `selection_key` (pipeline `score_vlm_crop_candidate`) instead of newest-N FIFO, so a later batch of weak detections cannot evict stronger earlier candidates.
- Updated `README.md` to describe score-based cache retention and to point to `visualize_vlm.py` for end-to-end cropper+VLM runs.
- Added `crop_cache_panel_width_unscaled()` so panel width matches `content_right` in `build_crop_cache_panel` (avoids extra dead space from the old `base_margin * 3` width estimate).

- Changed the crop cache from score-retained top-K storage back to a rolling sequence cache, so `config_vlm_crop_cache_size` now means the current round size rather than the best N seen so far.
- Tightened the retry loop so first dispatch only happens after a full round, retry clears the previous round, and a lost object after retry forces VLM to use the previous sent image instead of receiving a new late crop.
