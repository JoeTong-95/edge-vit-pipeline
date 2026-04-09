"""
Layer-local validation script for vehicle_state_layer.

Run from the project root:
    python src/vehicle-state-layer/test_vehicle_state_layer.py
"""

from vehicle_state_layer import (
    build_vehicle_state_layer_package,
    get_vehicle_state_record,
    initialize_vehicle_state_layer,
    update_vehicle_state_from_tracking,
    update_vehicle_state_from_vlm,
)


def _make_tracking_package(frame_id, track_ids, classes, statuses):
    return {
        "tracking_layer_frame_id": frame_id,
        "tracking_layer_track_id": track_ids,
        "tracking_layer_bbox": [[0, 0, 10, 10] for _ in track_ids],
        "tracking_layer_detector_class": classes,
        "tracking_layer_confidence": [0.9 for _ in track_ids],
        "tracking_layer_status": statuses,
    }


def main() -> None:
    initialize_vehicle_state_layer(prune_after_lost_frames=2)

    update_vehicle_state_from_tracking(
        _make_tracking_package(
            frame_id=1,
            track_ids=[101, 202],
            classes=["truck", "bus"],
            statuses=["new", "new"],
        )
    )

    record_101 = get_vehicle_state_record(101)
    assert record_101 is not None
    assert record_101["vehicle_state_layer_first_seen_frame"] == 1
    assert record_101["vehicle_state_layer_last_seen_frame"] == 1
    assert record_101["vehicle_state_layer_lost_frame_count"] == 0
    assert record_101["vehicle_state_layer_vehicle_class"] == "truck"
    assert record_101["vehicle_state_layer_vlm_called"] is False

    update_vehicle_state_from_tracking(
        _make_tracking_package(
            frame_id=2,
            track_ids=[101, 202],
            classes=["truck", "bus"],
            statuses=["active", "lost"],
        )
    )

    update_vehicle_state_from_vlm(
        {
            "vlm_layer_track_id": 101,
            "vlm_layer_label": "dump_truck",
            "vlm_layer_attributes": {
                "truck_type": "dump_truck",
                "wheel_count": 6,
                "estimated_weight_kg": "12000-18000",
            },
        }
    )

    record_101 = get_vehicle_state_record(101)
    record_202 = get_vehicle_state_record(202)
    assert record_101["vehicle_state_layer_last_seen_frame"] == 2
    assert record_101["vehicle_state_layer_truck_type"] == "dump_truck"
    assert record_101["vehicle_state_layer_semantic_tags"]["wheel_count"] == 6
    assert record_101["vehicle_state_layer_vlm_called"] is True
    assert record_202["vehicle_state_layer_lost_frame_count"] == 1

    update_vehicle_state_from_tracking(
        _make_tracking_package(
            frame_id=3,
            track_ids=[101, 202],
            classes=["truck", "bus"],
            statuses=["active", "lost"],
        )
    )
    update_vehicle_state_from_tracking(
        _make_tracking_package(
            frame_id=4,
            track_ids=[101, 202],
            classes=["truck", "bus"],
            statuses=["active", "lost"],
        )
    )

    assert get_vehicle_state_record(202) is None

    output_package = build_vehicle_state_layer_package()
    assert output_package["vehicle_state_layer_track_id"] == ["101"]
    assert output_package["vehicle_state_layer_vehicle_class"] == ["truck"]
    assert output_package["vehicle_state_layer_truck_type"] == ["dump_truck"]
    assert output_package["vehicle_state_layer_vlm_called"] == [True]

    print("vehicle_state_layer tests passed")


if __name__ == "__main__":
    main()
