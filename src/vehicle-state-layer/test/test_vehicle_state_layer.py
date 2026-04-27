"""
Layer-local validation script for vehicle_state_layer.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VEHICLE_STATE_LAYER_DIR = ROOT / 'src' / 'vehicle-state-layer'
VLM_LAYER_DIR = ROOT / 'src' / 'vlm-layer'
if str(VEHICLE_STATE_LAYER_DIR) not in sys.path:
    sys.path.insert(0, str(VEHICLE_STATE_LAYER_DIR))
if str(VLM_LAYER_DIR) not in sys.path:
    sys.path.insert(0, str(VLM_LAYER_DIR))

from layer import VLMRawResult, build_vlm_ack_package, build_vlm_layer_package
from vehicle_state_layer import (
    build_vehicle_state_layer_package,
    get_vehicle_state_record,
    initialize_vehicle_state_layer,
    update_vehicle_state_from_tracking,
    update_vehicle_state_from_vlm,
    update_vehicle_state_from_vlm_ack,
)


def _make_tracking_package(frame_id, track_ids, classes, statuses):
    return {
        'tracking_layer_frame_id': frame_id,
        'tracking_layer_track_id': track_ids,
        'tracking_layer_bbox': [[0, 0, 10, 10] for _ in track_ids],
        'tracking_layer_detector_class': classes,
        'tracking_layer_confidence': [0.9 for _ in track_ids],
        'tracking_layer_status': statuses,
    }


def main() -> None:
    initialize_vehicle_state_layer(prune_after_lost_frames=2)
    update_vehicle_state_from_tracking(_make_tracking_package(1, [101, 202], ['truck', 'bus'], ['new', 'new']))
    record_101 = get_vehicle_state_record(101)
    assert record_101['vehicle_state_layer_vlm_ack_status'] == 'not_requested'

    update_vehicle_state_from_vlm_ack(build_vlm_ack_package('101', 'retry_requested', 'crop_not_good', True))
    record_101 = get_vehicle_state_record(101)
    assert record_101['vehicle_state_layer_vlm_retry_requested'] is True

    update_vehicle_state_from_tracking(_make_tracking_package(2, [101, 202], ['truck', 'bus'], ['active', 'lost']))
    vlm_package = build_vlm_layer_package(VLMRawResult(
        vlm_layer_track_id='101',
        vlm_layer_query_type='vehicle_semantics_v1',
        vlm_layer_model_id='test-model',
        vlm_layer_raw_text='vehicle_type: dump_truck\naxle_count: 3\nack_status: accepted',
    ))
    update_vehicle_state_from_vlm(vlm_package)
    record_101 = get_vehicle_state_record(101)
    assert record_101['vehicle_state_layer_vlm_ack_status'] == 'accepted'
    assert record_101['vehicle_state_layer_terminal_status'] == 'done'

    update_vehicle_state_from_vlm_ack(build_vlm_ack_package('101', 'finalize_with_current', 'truck_left_scene', False))
    record_101 = get_vehicle_state_record(101)
    assert record_101['vehicle_state_layer_vlm_final_candidate_sent'] is True

    update_vehicle_state_from_tracking(_make_tracking_package(3, [101, 202], ['truck', 'bus'], ['active', 'lost']))
    update_vehicle_state_from_tracking(_make_tracking_package(4, [101, 202], ['truck', 'bus'], ['active', 'lost']))
    assert get_vehicle_state_record(202) is None

    update_vehicle_state_from_tracking(_make_tracking_package(5, [303], ['truck'], ['new']))
    not_truck_package = build_vlm_layer_package(VLMRawResult(
        vlm_layer_track_id='303',
        vlm_layer_query_type='vehicle_semantics_single_shot_v1',
        vlm_layer_model_id='test-model',
        vlm_layer_raw_text='{"is_target_vehicle": false, "vehicle_type": "sedan", "confidence": 0.91}',
    ))
    update_vehicle_state_from_vlm(not_truck_package)
    record_303 = get_vehicle_state_record(303)
    assert record_303['vehicle_state_layer_terminal_status'] == 'no'
    assert record_303['vehicle_state_layer_truck_type'] == 'unknown'

    output_package = build_vehicle_state_layer_package()
    assert output_package['vehicle_state_layer_terminal_status'] == ['done', 'no']
    print('vehicle_state_layer tests passed')


if __name__ == '__main__':
    main()
