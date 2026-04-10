"""
Layer-local validation script for vlm_frame_cropper_layer.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / 'src' / 'input-layer', ROOT / 'src' / 'vlm-layer'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from input_layer_package import InputLayerPackage
from layer import build_vlm_ack_package
from vlm_frame_cropper_layer import (
    build_vlm_dispatch_package,
    build_vlm_frame_cropper_package,
    build_vlm_frame_cropper_request_package,
    extract_vlm_object_crop,
    initialize_vlm_crop_cache,
    refresh_vlm_crop_cache_track_state,
    register_vlm_ack_package,
    resolve_source_frame,
    update_vlm_crop_cache,
)


def _make_input_layer_package(frame_id: int, image: np.ndarray) -> dict:
    return {
        'input_layer_frame_id': frame_id,
        'input_layer_timestamp': 2000.0 + frame_id,
        'input_layer_image': image,
        'input_layer_source_type': 'video',
        'input_layer_resolution': (image.shape[1], image.shape[0]),
    }


def _make_tracking_package(frame_id: int, confidence: float) -> dict:
    return {
        'tracking_layer_frame_id': frame_id,
        'tracking_layer_track_id': [101],
        'tracking_layer_bbox': [[10, 20, 40, 60]],
        'tracking_layer_detector_class': ['truck'],
        'tracking_layer_confidence': [confidence],
        'tracking_layer_status': ['active'],
    }


def _make_tracking_row(confidence: float, status: str) -> dict:
    return {
        'track_id': '101',
        'bbox': (10, 20, 40, 60),
        'detector_class': 'truck',
        'confidence': confidence,
        'status': status,
    }


def _cache_candidate(cache_state, frame_id: int, confidence: float, status: str = 'active'):
    frame = np.full((80, 90, 3), frame_id, dtype=np.uint8)
    input_package = InputLayerPackage(**_make_input_layer_package(frame_id, frame))
    tracking_package = _make_tracking_package(frame_id, confidence)
    request = build_vlm_frame_cropper_request_package(input_package, tracking_package, 0, f'tracking_status:{status}', True)
    assert resolve_source_frame(input_package, request).shape == frame.shape
    crop = extract_vlm_object_crop(input_package, request)
    package = build_vlm_frame_cropper_package(request, crop)
    update_vlm_crop_cache(cache_state, _make_tracking_row(confidence, status), package, frame_id, f'tracking_status:{status}')


def main() -> None:
    frame = np.arange(80 * 90 * 3, dtype=np.uint8).reshape((80, 90, 3))
    dict_input = _make_input_layer_package(5, frame)
    dataclass_input = InputLayerPackage(**dict_input)
    tracking_package = {
        'tracking_layer_frame_id': 5,
        'tracking_layer_track_id': [101],
        'tracking_layer_bbox': [[10, 20, 40, 60]],
        'tracking_layer_detector_class': ['truck'],
        'tracking_layer_confidence': [0.95],
        'tracking_layer_status': ['new'],
    }

    assert build_vlm_frame_cropper_request_package(dict_input, tracking_package, 0, 'new_track', False) is None
    request = build_vlm_frame_cropper_request_package(dataclass_input, tracking_package, 0, 'new_track', True)
    crop = extract_vlm_object_crop(dataclass_input, request)
    package = build_vlm_frame_cropper_package(request, crop)
    assert package['vlm_frame_cropper_layer_track_id'] == '101'

    cache_state = initialize_vlm_crop_cache(3)
    _cache_candidate(cache_state, 1, 0.70)
    _cache_candidate(cache_state, 2, 0.72)
    assert build_vlm_dispatch_package(cache_state, 101) is None

    # Losing the object before the first round is full should not dispatch anything.
    refresh_vlm_crop_cache_track_state(cache_state, _make_tracking_row(0.72, 'lost'), 2)
    assert build_vlm_dispatch_package(cache_state, 101) is None

    # Reacquire, finish the first round, and dispatch once.
    _cache_candidate(cache_state, 3, 0.90)
    first_dispatch = build_vlm_dispatch_package(cache_state, 101)
    assert first_dispatch is not None
    assert first_dispatch['vlm_dispatch_mode'] == 'initial_candidate'
    assert first_dispatch['vlm_dispatch_reason'] == 'cache_full'
    assert build_vlm_dispatch_package(cache_state, 101) is None

    # Retry clears the cache and requires a brand-new full round.
    register_vlm_ack_package(cache_state, build_vlm_ack_package('101', 'retry_requested', 'crop_not_good', True))
    track_cache = cache_state['track_caches']['101']
    assert track_cache['cached_crops'] == []
    assert track_cache['selected_crop'] is None
    _cache_candidate(cache_state, 4, 0.60)
    _cache_candidate(cache_state, 5, 0.65)
    assert build_vlm_dispatch_package(cache_state, 101) is None
    _cache_candidate(cache_state, 6, 0.99)
    retry_dispatch = build_vlm_dispatch_package(cache_state, 101)
    assert retry_dispatch is not None
    assert retry_dispatch['vlm_dispatch_mode'] == 'retry_candidate'

    # If retry is requested again but the object leaves before a new round fills,
    # cropper does not send a new image; it marks that VLM must use the previous sent crop.
    register_vlm_ack_package(cache_state, build_vlm_ack_package('101', 'retry_requested', 'still_not_good', True))
    _cache_candidate(cache_state, 7, 0.40)
    refresh_vlm_crop_cache_track_state(cache_state, _make_tracking_row(0.40, 'lost'), 7)
    assert build_vlm_dispatch_package(cache_state, 101) is None
    track_cache = cache_state['track_caches']['101']
    assert track_cache['vlm_previous_sent_must_be_used'] is True
    assert track_cache['vlm_last_dispatch_mode'] == 'use_previous_sent_final'
    assert track_cache['vlm_last_sent_package'] is not None

    # Cache is a rolling last-N sequence, not a top-K score store.
    cache_state2 = initialize_vlm_crop_cache(3)
    _cache_candidate(cache_state2, 10, 0.90)
    _cache_candidate(cache_state2, 11, 0.91)
    _cache_candidate(cache_state2, 12, 0.92)
    _cache_candidate(cache_state2, 13, 0.20)
    _cache_candidate(cache_state2, 14, 0.20)
    tc = cache_state2['track_caches']['101']
    assert [c['frame_id'] for c in tc['cached_crops']] == [12, 13, 14]
    assert tc['selected_crop']['frame_id'] == 12

    print('vlm_frame_cropper_layer tests passed')


if __name__ == '__main__':
    main()
