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
python src/vlm-frame-cropper-layer/test_vlm_frame_cropper_layer.py
python src/vlm-frame-cropper-layer/visualize_vlm_frame_cropper.py --show
```

For **cropper + live VLM inference + ack/retry logging**, use the VLM-layer helper (still uses this layer’s public API only):

```powershell
python src/vlm-layer/visualize_vlm.py --show
```

## Current Dispatch Loop

The cropper now follows the stricter loop used by the VLM visualizer:

- collect a rolling sequence of the last `config_vlm_crop_cache_size` crops for the current round
- score that round and dispatch one best crop only when the round is full
- if a track stays `lost` for `config_vlm_dead_after_lost_frames`, mark it `dead`
- if that `dead` track only collected a partial cache, still send the best available crop once
- if VLM replies `retry_requested`, clear the old round and collect a fresh round before re-evaluating
- if that retry round never fills because the object is lost, do not send a new crop; VLM must use the previous sent image
