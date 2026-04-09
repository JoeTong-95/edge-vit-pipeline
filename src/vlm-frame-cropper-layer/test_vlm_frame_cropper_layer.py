"""
Layer-local validation script for vlm_frame_cropper_layer.

Run from the project root:
    python src/vlm-frame-cropper-layer/test_vlm_frame_cropper_layer.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT_LAYER_DIR = ROOT / "src" / "input-layer"
if str(INPUT_LAYER_DIR) not in sys.path:
    sys.path.insert(0, str(INPUT_LAYER_DIR))

from input_layer_package import InputLayerPackage
from vlm_frame_cropper_layer import (
    build_vlm_frame_cropper_package,
    build_vlm_frame_cropper_request_package,
    extract_vlm_object_crop,
    resolve_source_frame,
)



def _make_input_layer_package(frame_id: int, image: np.ndarray) -> dict:
    return {
        "input_layer_frame_id": frame_id,
        "input_layer_timestamp": 2000.0 + frame_id,
        "input_layer_image": image,
        "input_layer_source_type": "video",
        "input_layer_resolution": (image.shape[1], image.shape[0]),
    }



def _make_tracking_layer_package(frame_id: int) -> dict:
    return {
        "tracking_layer_frame_id": frame_id,
        "tracking_layer_track_id": [101, 202],
        "tracking_layer_bbox": [
            [10, 20, 40, 60],
            [0, 0, 15, 10],
        ],
        "tracking_layer_detector_class": ["truck", "bus"],
        "tracking_layer_confidence": [0.95, 0.88],
        "tracking_layer_status": ["new", "active"],
    }



def main() -> None:
    frame = np.arange(80 * 90 * 3, dtype=np.uint8).reshape((80, 90, 3))
    dict_input_package = _make_input_layer_package(frame_id=5, image=frame)
    dataclass_input_package = InputLayerPackage(**dict_input_package)
    tracking_package = _make_tracking_layer_package(frame_id=5)

    disabled_request = build_vlm_frame_cropper_request_package(
        input_layer_package=dict_input_package,
        tracking_layer_package=tracking_package,
        track_index=0,
        vlm_frame_cropper_trigger_reason="new_track",
        config_vlm_enabled=False,
    )
    assert disabled_request is None

    request_package = build_vlm_frame_cropper_request_package(
        input_layer_package=dataclass_input_package,
        tracking_layer_package=tracking_package,
        track_index=0,
        vlm_frame_cropper_trigger_reason="new_track",
        config_vlm_enabled=True,
    )
    assert request_package["vlm_frame_cropper_frame_id"] == 5
    assert request_package["vlm_frame_cropper_track_id"] == "101"
    assert request_package["vlm_frame_cropper_bbox"] == (10, 20, 40, 60)
    assert request_package["vlm_frame_cropper_trigger_reason"] == "new_track"

    source_frame = resolve_source_frame(dataclass_input_package, request_package)
    assert source_frame.shape == frame.shape

    crop = extract_vlm_object_crop(dataclass_input_package, request_package)
    assert crop.shape == (40, 30, 3)
    np.testing.assert_array_equal(crop, frame[20:60, 10:40])

    cropper_package = build_vlm_frame_cropper_package(request_package, crop)
    assert cropper_package["vlm_frame_cropper_layer_track_id"] == "101"
    assert cropper_package["vlm_frame_cropper_layer_bbox"] == (10, 20, 40, 60)
    assert cropper_package["vlm_frame_cropper_layer_image"].shape == (40, 30, 3)

    print("vlm_frame_cropper_layer tests passed")


if __name__ == "__main__":
    main()
