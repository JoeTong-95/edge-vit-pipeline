"""
roi_layer.py

Layer 3: ROI

Purpose:
    Restrict processing to the relevant scene region before detection.

Public functions (from pipeline_layers_and_interactions.md):
    initialize_roi_layer
    update_roi_state
    apply_roi_to_frame
    build_roi_layer_package

Internal helpers implement the documented roi_cropper_node and
roi_discovery_node behavior.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


_state = {
    "initialized": False,
    "config_roi_enabled": False,
    "config_roi_vehicle_count_threshold": 0,
    "roi_layer_bounds": None,
    "roi_layer_locked": False,
    "roi_candidate_boxes": [],
}


def initialize_roi_layer(
    config_roi_enabled: bool,
    config_roi_vehicle_count_threshold: int,
) -> None:
    """Prepare ROI state from config values."""
    if config_roi_vehicle_count_threshold < 0:
        raise ValueError(
            "config_roi_vehicle_count_threshold must be non-negative."
        )

    _state["initialized"] = True
    _state["config_roi_enabled"] = bool(config_roi_enabled)
    _state["config_roi_vehicle_count_threshold"] = int(
        config_roi_vehicle_count_threshold
    )
    _state["roi_layer_bounds"] = None
    _state["roi_layer_locked"] = False
    _state["roi_candidate_boxes"] = []



def update_roi_state(
    input_layer_package: dict[str, Any],
    yolo_layer_detections: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Update ROI discovery state during startup.

    The ROI is collected from early-frame vehicle detections and locks once the
    configured count threshold is reached.
    """
    _assert_initialized()
    _validate_input_layer_package(input_layer_package)
    _validate_yolo_detections(yolo_layer_detections)

    if not _state["config_roi_enabled"] or _state["roi_layer_locked"]:
        return _snapshot_roi_state()

    collect_roi_candidate_boxes(yolo_layer_detections)

    threshold = _state["config_roi_vehicle_count_threshold"]
    if threshold == 0:
        computed_bounds = _full_frame_bounds(input_layer_package["input_layer_image"])
        lock_roi_bounds(computed_bounds)
        return _snapshot_roi_state()

    if len(_state["roi_candidate_boxes"]) >= threshold:
        computed_bounds = compute_roi_bounds(
            frame_image=input_layer_package["input_layer_image"]
        )
        lock_roi_bounds(computed_bounds)

    return _snapshot_roi_state()



def apply_roi_to_frame(input_layer_package: dict[str, Any]) -> np.ndarray:
    """Apply the active ROI to the input frame."""
    _assert_initialized()
    _validate_input_layer_package(input_layer_package)

    frame_image = input_layer_package["input_layer_image"]
    if not _state["config_roi_enabled"] or not _state["roi_layer_locked"]:
        return pass_through_full_frame(frame_image)

    return crop_frame_to_roi(frame_image, _state["roi_layer_bounds"])



def build_roi_layer_package(
    input_layer_package: dict[str, Any],
    roi_layer_image: np.ndarray | None = None,
) -> dict[str, Any]:
    """Create the ROI package for downstream detection."""
    _assert_initialized()
    _validate_input_layer_package(input_layer_package)

    if roi_layer_image is None:
        roi_layer_image = apply_roi_to_frame(input_layer_package)

    roi_bounds = _state["roi_layer_bounds"]
    if roi_bounds is None:
        roi_bounds = _full_frame_bounds(input_layer_package["input_layer_image"])

    return {
        "roi_layer_frame_id": input_layer_package["input_layer_frame_id"],
        "roi_layer_timestamp": input_layer_package["input_layer_timestamp"],
        "roi_layer_image": roi_layer_image,
        "roi_layer_bounds": tuple(int(value) for value in roi_bounds),
        "roi_layer_enabled": _state["config_roi_enabled"],
        "roi_layer_locked": _state["roi_layer_locked"],
    }



def crop_frame_to_roi(
    frame_image: np.ndarray,
    roi_bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Crop the frame using the active ROI bounds."""
    x_min, y_min, x_max, y_max = _normalize_bounds(
        roi_bounds=roi_bounds,
        frame_image=frame_image,
    )
    return frame_image[y_min:y_max, x_min:x_max].copy()



def pass_through_full_frame(frame_image: np.ndarray) -> np.ndarray:
    """Return the full frame when ROI is disabled or not yet locked."""
    return frame_image.copy()



def collect_roi_candidate_boxes(
    yolo_layer_detections: list[dict[str, Any]],
) -> None:
    """Store candidate vehicle detections during startup."""
    for detection in yolo_layer_detections:
        bbox = detection["yolo_detection_bbox"]
        _state["roi_candidate_boxes"].append(tuple(float(value) for value in bbox))



def compute_roi_bounds(frame_image: np.ndarray) -> tuple[int, int, int, int]:
    """Compute ROI bounds from collected detections."""
    if not _state["roi_candidate_boxes"]:
        return _full_frame_bounds(frame_image)

    candidate_array = np.array(_state["roi_candidate_boxes"], dtype=np.float32)
    x_min = int(np.floor(candidate_array[:, 0].min()))
    y_min = int(np.floor(candidate_array[:, 1].min()))
    x_max = int(np.ceil(candidate_array[:, 2].max()))
    y_max = int(np.ceil(candidate_array[:, 3].max()))
    return _normalize_bounds((x_min, y_min, x_max, y_max), frame_image)



def lock_roi_bounds(roi_bounds: tuple[int, int, int, int]) -> None:
    """Mark ROI discovery as complete and freeze the active ROI."""
    _state["roi_layer_bounds"] = tuple(int(value) for value in roi_bounds)
    _state["roi_layer_locked"] = True



def _assert_initialized() -> None:
    if not _state["initialized"]:
        raise RuntimeError(
            "roi_layer has not been initialized. Call initialize_roi_layer() first."
        )



def _snapshot_roi_state() -> dict[str, Any]:
    return {
        "roi_layer_bounds": deepcopy(_state["roi_layer_bounds"]),
        "roi_layer_locked": _state["roi_layer_locked"],
        "roi_candidate_box_count": len(_state["roi_candidate_boxes"]),
        "roi_layer_enabled": _state["config_roi_enabled"],
    }



def _validate_input_layer_package(input_layer_package: dict[str, Any]) -> None:
    required_fields = [
        "input_layer_frame_id",
        "input_layer_timestamp",
        "input_layer_image",
        "input_layer_source_type",
        "input_layer_resolution",
    ]
    missing_fields = [
        field_name for field_name in required_fields if field_name not in input_layer_package
    ]
    if missing_fields:
        raise ValueError(
            "input_layer_package is missing required fields: "
            + ", ".join(missing_fields)
        )



def _validate_yolo_detections(yolo_layer_detections: list[dict[str, Any]]) -> None:
    for detection in yolo_layer_detections:
        required_fields = [
            "yolo_detection_bbox",
            "yolo_detection_class",
            "yolo_detection_confidence",
        ]
        missing_fields = [
            field_name for field_name in required_fields if field_name not in detection
        ]
        if missing_fields:
            raise ValueError(
                "yolo detection is missing required fields: "
                + ", ".join(missing_fields)
            )



def _normalize_bounds(
    roi_bounds: tuple[int | float, int | float, int | float, int | float],
    frame_image: np.ndarray,
) -> tuple[int, int, int, int]:
    frame_height, frame_width = frame_image.shape[:2]

    x_min = max(0, min(int(np.floor(roi_bounds[0])), frame_width - 1))
    y_min = max(0, min(int(np.floor(roi_bounds[1])), frame_height - 1))
    x_max = max(x_min + 1, min(int(np.ceil(roi_bounds[2])), frame_width))
    y_max = max(y_min + 1, min(int(np.ceil(roi_bounds[3])), frame_height))

    return (x_min, y_min, x_max, y_max)



def _full_frame_bounds(frame_image: np.ndarray) -> tuple[int, int, int, int]:
    frame_height, frame_width = frame_image.shape[:2]
    return (0, 0, frame_width, frame_height)


__all__ = [
    "apply_roi_to_frame",
    "build_roi_layer_package",
    "collect_roi_candidate_boxes",
    "compute_roi_bounds",
    "crop_frame_to_roi",
    "initialize_roi_layer",
    "lock_roi_bounds",
    "pass_through_full_frame",
    "update_roi_state",
]
