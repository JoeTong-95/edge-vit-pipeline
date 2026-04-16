# VLM Frame Cropper Layer

This layer prepares object-level image crops for semantic reasoning.

## Public API

- `build_vlm_frame_cropper_request_package`
- `extract_vlm_object_crop`
- `build_vlm_frame_cropper_package`

## Ownership

- Builds one crop request for one tracked object.
- Resolves the source frame from the input layer package.
- Extracts and validates the crop before handing it to the VLM layer.
- Expands each tracked bbox with a small context margin before slicing so the
  crop keeps a bit more of the vehicle and surrounding scene instead of
  clipping tightly to the detector box.
- Maintains a local per-track crop cache so the best candidate can be chosen before VLM is called.
- Owns the one-shot dispatch loop for crop selection, retry, and final-fallback delivery.

## Runtime Dispatch Loop

The implemented runtime policy is:

1. Collect crop candidates for one `track_id` until the cache reaches `config_vlm_crop_cache_size`.
2. Retain up to that many candidates by **score** (`score_vlm_crop_candidate`), so good crops from earlier frames are not evicted by a burst of weaker detections; then rank and send exactly one best crop to VLM (`select_best_vlm_crop_candidate`).
3. Do not send that track again unless VLM sends an ack with `vlm_ack_status="retry_requested"`.
4. If retry is requested and the truck is still `new` or `active`, wait for a newer or better candidate and resend the best one.
5. If a track stays `lost` for `config_vlm_dead_after_lost_frames` consecutive updates, the cropper marks it `dead`.
6. If a `dead` track never filled the configured cache size, the cropper still sends the best available partial cache instead of dropping the track entirely.
7. After an `accepted` ack, VLM decides whether the terminal state is `no` (rejected as not one of the flagged labels) or `done` (accepted flagged-label JSON classification).
8. After an `accepted` or `finalize_with_current` ack, that track is finalized and no more crops are dispatched.

## Crop Selection

The selector lives in `score_vlm_crop_candidate` inside `src/vlm-frame-cropper-layer/vlm_frame_cropper_layer.py`.

If you want to tune the formula, edit these constants near the top of that file:

- `CROP_SELECTION_CONFIDENCE_WEIGHT`
- `CROP_SELECTION_AREA_WEIGHT`
- `CROP_SELECTION_RECENCY_WEIGHT`
- `CROP_SELECTION_AREA_NORMALIZER`
- `CROP_SELECTION_FRAME_NORMALIZER`

## Ack Handling

The cropper layer expects VLM-side acknowledgement packages with these fields:

- `vlm_ack_track_id`
- `vlm_ack_status`
- `vlm_ack_reason`
- `vlm_ack_retry_requested`

Supported ack statuses:

- `accepted`
- `retry_requested`
- `finalize_with_current`

The cropper visualizer now also exposes the local terminal state in practice:

- `collecting`: still filling cache or waiting for first dispatch
- `no`: VLM rejected the track as not one of the flagged labels
- `dead`: lost too long before a successful terminal semantic answer
- `done`: accepted truck JSON has been recorded

## Quick Start

From the project root:

```powershell
python src/vlm-frame-cropper-layer/test/test_vlm_frame_cropper_layer.py
python src/vlm-frame-cropper-layer/util/visualize_vlm_frame_cropper.py --show
```

For **cropper + live VLM inference + ack/retry logging**, use the VLM-layer helper (still uses this layer’s public API only):

```powershell
python src/vlm-layer/util/visualize_vlm.py --show
```

## Current Dispatch Loop

The cropper now follows the stricter loop used by the VLM visualizer:

- collect a rolling sequence of the last `config_vlm_crop_cache_size` crops for the current round
- score that round and dispatch one best crop only when the round is full
- if a track stays `lost` for `config_vlm_dead_after_lost_frames`, mark it `dead`
- if that `dead` track only collected a partial cache, still send the best available crop once
- if VLM replies `retry_requested`, clear the old round and collect a fresh round before re-evaluating
- if that retry round never fills because the object is lost, do not send a new crop; VLM must use the previous sent image

## Changes

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

## 2026-04-10

- Added `config_vlm_dead_after_lost_frames` support to the crop cache state so a
  track can become terminal `dead` after a configurable consecutive-lost streak.
- Added dead-track partial dispatch behavior: if the first cache round never
  reaches `config_vlm_crop_cache_size`, the cropper now still sends the best
  available crop once the track is dead.
- Added local terminal states `collecting`, `dead`, and `done` for cropper-side
  visualization and downstream orchestration.
- Updated the cropper visualizer so dead and done tracks render with explicit
  statuses instead of only `new`/`active`/`lost`.
- Split VLM rejection from cropper death: the visualizer and local terminal
  state now show `no` for VLM label rejection, while `dead` remains the
  lost-threshold path.
