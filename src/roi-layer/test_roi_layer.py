"""
Layer-local validation script for roi_layer.

Run from the project root:
    python src/roi-layer/test_roi_layer.py
"""

import numpy as np

from roi_layer import (
    apply_roi_to_frame,
    build_roi_layer_package,
    initialize_roi_layer,
    update_roi_state,
)



def _make_input_layer_package(frame_id: int, image: np.ndarray) -> dict:
    return {
        "input_layer_frame_id": frame_id,
        "input_layer_timestamp": 1000.0 + frame_id,
        "input_layer_image": image,
        "input_layer_source_type": "video",
        "input_layer_resolution": (image.shape[1], image.shape[0]),
    }



def _make_detection(x_min, y_min, x_max, y_max, class_name="truck", confidence=0.9):
    return {
        "yolo_detection_bbox": [x_min, y_min, x_max, y_max],
        "yolo_detection_class": class_name,
        "yolo_detection_confidence": confidence,
    }



def main() -> None:
    frame = np.arange(100 * 120 * 3, dtype=np.uint8).reshape((100, 120, 3))
    input_package = _make_input_layer_package(frame_id=1, image=frame)

    initialize_roi_layer(config_roi_enabled=False, config_roi_vehicle_count_threshold=2)
    passthrough_image = apply_roi_to_frame(input_package)
    passthrough_package = build_roi_layer_package(input_package, passthrough_image)
    assert passthrough_image.shape == frame.shape
    assert passthrough_package["roi_layer_enabled"] is False
    assert passthrough_package["roi_layer_locked"] is False
    assert passthrough_package["roi_layer_bounds"] == (0, 0, 120, 100)

    initialize_roi_layer(config_roi_enabled=True, config_roi_vehicle_count_threshold=2)
    state_after_first_detection = update_roi_state(
        input_package,
        [_make_detection(10, 15, 50, 70)],
    )
    assert state_after_first_detection["roi_layer_locked"] is False
    assert state_after_first_detection["roi_candidate_box_count"] == 1

    state_after_second_detection = update_roi_state(
        input_package,
        [_make_detection(20, 10, 80, 90)],
    )
    assert state_after_second_detection["roi_layer_locked"] is True
    assert state_after_second_detection["roi_layer_bounds"] == (10, 10, 80, 90)

    cropped_image = apply_roi_to_frame(input_package)
    assert cropped_image.shape == (80, 70, 3)

    roi_package = build_roi_layer_package(input_package, cropped_image)
    assert roi_package["roi_layer_frame_id"] == 1
    assert roi_package["roi_layer_timestamp"] == 1001.0
    assert roi_package["roi_layer_bounds"] == (10, 10, 80, 90)
    assert roi_package["roi_layer_enabled"] is True
    assert roi_package["roi_layer_locked"] is True
    assert roi_package["roi_layer_image"].shape == (80, 70, 3)

    print("roi_layer tests passed")


if __name__ == "__main__":
    main()
