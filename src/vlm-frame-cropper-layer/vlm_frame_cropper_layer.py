"""
vlm_frame_cropper_layer.py

Layer 7: VLM Frame Cropper

Purpose:
    Prepare object-level image crops for semantic reasoning.

Public functions (from pipeline_layers_and_interactions.md):
    build_vlm_frame_cropper_request_package
    extract_vlm_object_crop
    build_vlm_frame_cropper_package

Internal helpers implement the documented vlm_frame_cropper_node behavior.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np



def build_vlm_frame_cropper_request_package(
    input_layer_package: Any,
    tracking_layer_package: dict[str, Any],
    track_index: int,
    vlm_frame_cropper_trigger_reason: str,
    config_vlm_enabled: bool = True,
) -> dict[str, Any] | None:
    """
    Create the request package for object cropping.

    Returns ``None`` when the VLM path is disabled.
    """
    _validate_input_layer_package(input_layer_package)
    _validate_tracking_layer_package(tracking_layer_package)

    if not config_vlm_enabled:
        return None

    track_count = len(tracking_layer_package["tracking_layer_track_id"])
    if track_index < 0 or track_index >= track_count:
        raise IndexError("track_index is out of range for tracking_layer_package.")

    return {
        "vlm_frame_cropper_frame_id": _package_get(input_layer_package, "input_layer_frame_id"),
        "vlm_frame_cropper_track_id": str(
            tracking_layer_package["tracking_layer_track_id"][track_index]
        ),
        "vlm_frame_cropper_bbox": tuple(
            tracking_layer_package["tracking_layer_bbox"][track_index]
        ),
        "vlm_frame_cropper_trigger_reason": vlm_frame_cropper_trigger_reason,
    }



def extract_vlm_object_crop(
    input_layer_package: Any,
    vlm_frame_cropper_request_package: dict[str, Any],
) -> np.ndarray:
    """Cut the target object crop from the source frame."""
    source_frame = resolve_source_frame(
        input_layer_package=input_layer_package,
        vlm_frame_cropper_request_package=vlm_frame_cropper_request_package,
    )
    object_crop = crop_object_from_frame(
        source_frame=source_frame,
        vlm_frame_cropper_bbox=vlm_frame_cropper_request_package[
            "vlm_frame_cropper_bbox"
        ],
    )
    validate_crop_result(object_crop)
    return object_crop



def build_vlm_frame_cropper_package(
    vlm_frame_cropper_request_package: dict[str, Any],
    vlm_object_crop: np.ndarray,
) -> dict[str, Any]:
    """Create the crop package for VLM inference."""
    _validate_request_package(vlm_frame_cropper_request_package)
    validate_crop_result(vlm_object_crop)

    return {
        "vlm_frame_cropper_layer_track_id": deepcopy(
            vlm_frame_cropper_request_package["vlm_frame_cropper_track_id"]
        ),
        "vlm_frame_cropper_layer_image": vlm_object_crop.copy(),
        "vlm_frame_cropper_layer_bbox": tuple(
            int(value)
            for value in vlm_frame_cropper_request_package["vlm_frame_cropper_bbox"]
        ),
    }



def resolve_source_frame(
    input_layer_package: Any,
    vlm_frame_cropper_request_package: dict[str, Any],
) -> np.ndarray:
    """Retrieve the source frame associated with the crop request."""
    _validate_input_layer_package(input_layer_package)
    _validate_request_package(vlm_frame_cropper_request_package)

    if (
        _package_get(input_layer_package, "input_layer_frame_id")
        != vlm_frame_cropper_request_package["vlm_frame_cropper_frame_id"]
    ):
        raise ValueError(
            "input_layer_frame_id does not match vlm_frame_cropper_frame_id."
        )

    return _package_get(input_layer_package, "input_layer_image")



def crop_object_from_frame(
    source_frame: np.ndarray,
    vlm_frame_cropper_bbox: tuple[int | float, int | float, int | float, int | float],
) -> np.ndarray:
    """Extract the object image using the requested bounding box."""
    x_min, y_min, x_max, y_max = _normalize_bbox(
        bbox=vlm_frame_cropper_bbox,
        frame_image=source_frame,
    )
    return source_frame[y_min:y_max, x_min:x_max].copy()



def validate_crop_result(vlm_object_crop: np.ndarray) -> None:
    """Check that the crop is usable before VLM inference."""
    if not isinstance(vlm_object_crop, np.ndarray):
        raise TypeError("vlm_object_crop must be a numpy ndarray.")
    if vlm_object_crop.size == 0:
        raise ValueError("vlm_object_crop is empty.")
    if len(vlm_object_crop.shape) < 2:
        raise ValueError("vlm_object_crop must have at least 2 dimensions.")
    if vlm_object_crop.shape[0] <= 0 or vlm_object_crop.shape[1] <= 0:
        raise ValueError("vlm_object_crop has invalid spatial dimensions.")



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



def _validate_tracking_layer_package(tracking_layer_package: dict[str, Any]) -> None:
    required_fields = [
        "tracking_layer_frame_id",
        "tracking_layer_track_id",
        "tracking_layer_bbox",
        "tracking_layer_detector_class",
        "tracking_layer_confidence",
        "tracking_layer_status",
    ]
    missing_fields = [
        field_name for field_name in required_fields if field_name not in tracking_layer_package
    ]
    if missing_fields:
        raise ValueError(
            "tracking_layer_package is missing required fields: "
            + ", ".join(missing_fields)
        )

    list_lengths = {
        field_name: len(tracking_layer_package[field_name])
        for field_name in required_fields
        if field_name != "tracking_layer_frame_id"
    }
    if len(set(list_lengths.values())) > 1:
        raise ValueError(
            "tracking_layer_package list fields must all have the same length."
        )



def _validate_request_package(vlm_frame_cropper_request_package: dict[str, Any]) -> None:
    required_fields = [
        "vlm_frame_cropper_frame_id",
        "vlm_frame_cropper_track_id",
        "vlm_frame_cropper_bbox",
        "vlm_frame_cropper_trigger_reason",
    ]
    missing_fields = [
        field_name
        for field_name in required_fields
        if field_name not in vlm_frame_cropper_request_package
    ]
    if missing_fields:
        raise ValueError(
            "vlm_frame_cropper_request_package is missing required fields: "
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



def _normalize_bbox(
    bbox: tuple[int | float, int | float, int | float, int | float],
    frame_image: np.ndarray,
) -> tuple[int, int, int, int]:
    frame_height, frame_width = frame_image.shape[:2]

    x_min = max(0, min(int(np.floor(bbox[0])), frame_width - 1))
    y_min = max(0, min(int(np.floor(bbox[1])), frame_height - 1))
    x_max = max(x_min + 1, min(int(np.ceil(bbox[2])), frame_width))
    y_max = max(y_min + 1, min(int(np.ceil(bbox[3])), frame_height))

    return (x_min, y_min, x_max, y_max)


__all__ = [
    "build_vlm_frame_cropper_package",
    "build_vlm_frame_cropper_request_package",
    "crop_object_from_frame",
    "extract_vlm_object_crop",
    "resolve_source_frame",
    "validate_crop_result",
]
