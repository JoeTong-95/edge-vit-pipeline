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

ROI_CANDIDATE_IOU_DUPLICATE_THRESHOLD = 0.5


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
    input_layer_package: Any,
    yolo_layer_detections: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Update ROI discovery state during startup.

    The ROI is collected from early-frame vehicle detections and locks once the
    configured count threshold is reached. Repeated detections of the same
    startup vehicle are deduplicated locally inside this layer so the threshold
    behaves more like a unique-ish vehicle count than a raw frame count.
    """
    _assert_initialized()
    _validate_input_layer_package(input_layer_package)
    _validate_yolo_detections(yolo_layer_detections)

    if not _state["config_roi_enabled"] or _state["roi_layer_locked"]:
        return _snapshot_roi_state()

    collect_roi_candidate_boxes(yolo_layer_detections)

    frame_image = _package_get(input_layer_package, "input_layer_image")
    threshold = _state["config_roi_vehicle_count_threshold"]
    if threshold == 0:
        lock_roi_bounds(_full_frame_bounds(frame_image))
        return _snapshot_roi_state()

    if len(_state["roi_candidate_boxes"]) >= threshold:
        computed_bounds = compute_roi_bounds(frame_image=frame_image)
        lock_roi_bounds(computed_bounds)

    return _snapshot_roi_state()



def apply_roi_to_frame(input_layer_package: Any) -> np.ndarray:
    """Apply the active ROI to the input frame."""
    _assert_initialized()
    _validate_input_layer_package(input_layer_package)

    frame_image = _package_get(input_layer_package, "input_layer_image")
    if not _state["config_roi_enabled"] or not _state["roi_layer_locked"]:
        return pass_through_full_frame(frame_image)

    return crop_frame_to_roi(frame_image, _state["roi_layer_bounds"])



def build_roi_layer_package(
    input_layer_package: Any,
    roi_layer_image: np.ndarray | None = None,
) -> dict[str, Any]:
    """Create the ROI package for downstream detection."""
    _assert_initialized()
    _validate_input_layer_package(input_layer_package)

    if roi_layer_image is None:
        roi_layer_image = apply_roi_to_frame(input_layer_package)

    frame_image = _package_get(input_layer_package, "input_layer_image")
    roi_bounds = _state["roi_layer_bounds"]
    if roi_bounds is None:
        roi_bounds = _full_frame_bounds(frame_image)

    return {
        "roi_layer_frame_id": _package_get(input_layer_package, "input_layer_frame_id"),
        "roi_layer_timestamp": _package_get(input_layer_package, "input_layer_timestamp"),
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
    """Store unique-ish candidate vehicle detections during startup."""
    for detection in yolo_layer_detections:
        bbox = tuple(float(value) for value in detection["yolo_detection_bbox"])
        if _is_duplicate_candidate_box(bbox):
            continue
        _state["roi_candidate_boxes"].append(bbox)



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


def _is_duplicate_candidate_box(
    candidate_bbox: tuple[float, float, float, float],
) -> bool:
    for existing_bbox in _state["roi_candidate_boxes"]:
        if _bbox_iou(candidate_bbox, existing_bbox) >= ROI_CANDIDATE_IOU_DUPLICATE_THRESHOLD:
            return True
    return False


def _bbox_iou(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area



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



def _validate_input_layer_package(input_layer_package: Any) -> None:
    required_fields = [
        "input_layer_frame_id",
        "input_layer_timestamp",
        "input_layer_image",
        "input_layer_source_type",
        "input_layer_resolution",
    ]
    _validate_required_fields(
        package=input_layer_package,
        required_fields=required_fields,
        package_name="input_layer_package",
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



def _validate_required_fields(
    package: Any,
    required_fields: list[str],
    package_name: str,
) -> None:
    missing_fields = [
        field_name for field_name in required_fields if not _package_has(package, field_name)
    ]
    if missing_fields:
        raise ValueError(
            f"{package_name} is missing required fields: {', '.join(missing_fields)}"
        )



def _package_has(package: Any, field_name: str) -> bool:
    return (isinstance(package, dict) and field_name in package) or hasattr(package, field_name)



def _package_get(package: Any, field_name: str) -> Any:
    if isinstance(package, dict):
        return package[field_name]
    return getattr(package, field_name)



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
