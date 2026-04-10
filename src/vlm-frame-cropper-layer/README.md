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
- Maintains a local per-track crop cache so the best candidate can be chosen before VLM is called.
- Owns the one-shot dispatch loop for crop selection, retry, and final-fallback delivery.

## Runtime Dispatch Loop

The implemented runtime policy is:

1. Collect crop candidates for one `track_id` until the cache reaches `config_vlm_crop_cache_size`.
2. Retain up to that many candidates by **score** (`score_vlm_crop_candidate`), so good crops from earlier frames are not evicted by a burst of weaker detections; then rank and send exactly one best crop to VLM (`select_best_vlm_crop_candidate`).
3. Do not send that track again unless VLM sends an ack with `vlm_ack_status="retry_requested"`.
4. If retry is requested and the truck is still `new` or `active`, wait for a newer or better candidate and resend the best one.
5. If retry is requested but the truck is `lost`, send one final best-available crop with `vlm_dispatch_mode="final_available_candidate"`.
6. After an `accepted` or `finalize_with_current` ack, that track is finalized and no more crops are dispatched.

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
- if VLM replies `retry_requested`, clear the old round and collect a fresh round before re-evaluating
- if that retry round never fills because the object is lost, do not send a new crop; VLM must use the previous sent image
