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
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

CROP_SELECTION_CONFIDENCE_WEIGHT = 1.0
CROP_SELECTION_AREA_WEIGHT = 0.35
CROP_SELECTION_RECENCY_WEIGHT = 0.05
CROP_SELECTION_AREA_NORMALIZER = 50000.0
CROP_SELECTION_FRAME_NORMALIZER = 10000.0
VLM_CROP_CONTEXT_PADDING_RATIO = 0.12

_ALLOWED_VLM_ACK_STATUSES = {"accepted", "retry_requested", "finalize_with_current"}
_NO_ACK_REASON = "no"


def build_vlm_frame_cropper_request_package(
    input_layer_package: Any,
    tracking_layer_package: dict[str, Any],
    track_index: int,
    vlm_frame_cropper_trigger_reason: str,
    config_vlm_enabled: bool = True,
) -> dict[str, Any] | None:
    _validate_input_layer_package(input_layer_package)
    _validate_tracking_layer_package(tracking_layer_package)
    if not config_vlm_enabled:
        return None
    track_count = len(tracking_layer_package["tracking_layer_track_id"])
    if track_index < 0 or track_index >= track_count:
        raise IndexError("track_index is out of range for tracking_layer_package.")
    return {
        "vlm_frame_cropper_frame_id": _package_get(input_layer_package, "input_layer_frame_id"),
        "vlm_frame_cropper_track_id": str(tracking_layer_package["tracking_layer_track_id"][track_index]),
        "vlm_frame_cropper_bbox": tuple(tracking_layer_package["tracking_layer_bbox"][track_index]),
        "vlm_frame_cropper_trigger_reason": vlm_frame_cropper_trigger_reason,
    }


def extract_vlm_object_crop(input_layer_package: Any, vlm_frame_cropper_request_package: dict[str, Any]) -> np.ndarray:
    source_frame = resolve_source_frame(input_layer_package=input_layer_package, vlm_frame_cropper_request_package=vlm_frame_cropper_request_package)
    object_crop = crop_object_from_frame(source_frame=source_frame, vlm_frame_cropper_bbox=vlm_frame_cropper_request_package["vlm_frame_cropper_bbox"])
    validate_crop_result(object_crop)
    return object_crop


def build_vlm_frame_cropper_package(vlm_frame_cropper_request_package: dict[str, Any], vlm_object_crop: np.ndarray) -> dict[str, Any]:
    _validate_request_package(vlm_frame_cropper_request_package)
    validate_crop_result(vlm_object_crop)
    return {
        "vlm_frame_cropper_layer_track_id": deepcopy(vlm_frame_cropper_request_package["vlm_frame_cropper_track_id"]),
        "vlm_frame_cropper_layer_image": vlm_object_crop.copy(),
        "vlm_frame_cropper_layer_bbox": tuple(int(value) for value in vlm_frame_cropper_request_package["vlm_frame_cropper_bbox"]),
    }


def initialize_vlm_crop_cache(config_vlm_crop_cache_size: int = 5, config_vlm_dead_after_lost_frames: int = 3) -> dict[str, Any]:
    cache_size = int(config_vlm_crop_cache_size)
    if cache_size <= 0:
        raise ValueError("config_vlm_crop_cache_size must be greater than 0.")
    dead_after_lost_frames = int(config_vlm_dead_after_lost_frames)
    if dead_after_lost_frames <= 0:
        raise ValueError("config_vlm_dead_after_lost_frames must be greater than 0.")
    return {
        "config_vlm_crop_cache_size": cache_size,
        "config_vlm_dead_after_lost_frames": dead_after_lost_frames,
        "track_caches": {},
    }


def refresh_vlm_crop_cache_track_state(vlm_crop_cache_state: dict[str, Any], tracking_layer_row: dict[str, Any], vlm_frame_cropper_frame_id: int) -> dict[str, Any]:
    _validate_vlm_crop_cache_state(vlm_crop_cache_state)
    _validate_tracking_layer_row(tracking_layer_row)
    track_cache = _ensure_track_cache_entry(vlm_crop_cache_state=vlm_crop_cache_state, track_id=tracking_layer_row["track_id"])
    track_cache["last_status"] = str(tracking_layer_row["status"])
    track_cache["last_frame_id"] = int(vlm_frame_cropper_frame_id)
    track_cache["detector_class"] = str(tracking_layer_row["detector_class"])
    if track_cache["last_status"] == "lost":
        track_cache["lost_frame_streak"] += 1
    else:
        track_cache["lost_frame_streak"] = 0

    dead_after_lost_frames = vlm_crop_cache_state["config_vlm_dead_after_lost_frames"]
    if track_cache["lost_frame_streak"] >= dead_after_lost_frames:
        track_cache["vlm_dead"] = True
        if track_cache["vlm_terminal_state"] == "collecting":
            track_cache["vlm_terminal_state"] = "dead"

    if (
        track_cache["vlm_retry_requested"]
        and track_cache["last_status"] == "lost"
        and track_cache["vlm_sent_count"] > 0
        and track_cache["lost_frame_streak"] >= dead_after_lost_frames
    ):
        track_cache["vlm_retry_requested"] = False
        track_cache["vlm_request_in_flight"] = False
        track_cache["vlm_finalized"] = True
        track_cache["vlm_previous_sent_must_be_used"] = True
        track_cache["vlm_last_dispatch_mode"] = "use_previous_sent_final"
        track_cache["vlm_last_dispatch_reason"] = "retry_requested_lost_use_previous_sent"
    return track_cache


def update_vlm_crop_cache(vlm_crop_cache_state: dict[str, Any], tracking_layer_row: dict[str, Any], vlm_frame_cropper_layer_package: dict[str, Any], vlm_frame_cropper_frame_id: int, vlm_frame_cropper_trigger_reason: str) -> dict[str, Any]:
    """Append one crop for this track; cache is the rolling last-N sequence for the current selection round."""
    _validate_vlm_crop_cache_state(vlm_crop_cache_state)
    _validate_tracking_layer_row(tracking_layer_row)
    _validate_cropper_layer_package(vlm_frame_cropper_layer_package)
    track_cache = refresh_vlm_crop_cache_track_state(vlm_crop_cache_state=vlm_crop_cache_state, tracking_layer_row=tracking_layer_row, vlm_frame_cropper_frame_id=vlm_frame_cropper_frame_id)
    if str(tracking_layer_row["status"]) == "lost" or track_cache["vlm_finalized"] or track_cache["vlm_dead"]:
        return track_cache

    cached_crop = {
        "frame_id": int(vlm_frame_cropper_frame_id),
        "track_id": str(vlm_frame_cropper_layer_package["vlm_frame_cropper_layer_track_id"]),
        "bbox": tuple(int(value) for value in vlm_frame_cropper_layer_package["vlm_frame_cropper_layer_bbox"]),
        "status": str(tracking_layer_row["status"]),
        "detector_class": str(tracking_layer_row["detector_class"]),
        "confidence": float(tracking_layer_row["confidence"]),
        "trigger_reason": str(vlm_frame_cropper_trigger_reason),
        "crop": vlm_frame_cropper_layer_package["vlm_frame_cropper_layer_image"].copy(),
    }
    cached_crop["selection_key"] = score_vlm_crop_candidate(cached_crop)
    track_cache["cached_crops"].append(cached_crop)
    cache_size = vlm_crop_cache_state["config_vlm_crop_cache_size"]
    track_cache["cached_crops"] = track_cache["cached_crops"][-cache_size:]
    track_cache["selected_crop"] = select_best_vlm_crop_candidate(track_cache["cached_crops"])
    return track_cache


def build_vlm_dispatch_package(vlm_crop_cache_state: dict[str, Any], vlm_frame_cropper_track_id: str | int) -> dict[str, Any] | None:
    _validate_vlm_crop_cache_state(vlm_crop_cache_state)
    track_cache = _get_track_cache(vlm_crop_cache_state, vlm_frame_cropper_track_id)
    if track_cache is None or track_cache["selected_crop"] is None or track_cache["vlm_finalized"]:
        return None
    if track_cache["vlm_request_in_flight"]:
        return None

    cache_size = vlm_crop_cache_state["config_vlm_crop_cache_size"]
    cached_crop_count = len(track_cache["cached_crops"])
    selected_crop = track_cache["selected_crop"]
    dispatch_mode = None
    dispatch_reason = None

    if track_cache["vlm_sent_count"] == 0 and track_cache["vlm_dead"]:
        dispatch_mode = "dead_best_available"
        dispatch_reason = "lost_streak_threshold_reached_partial_cache"
    elif track_cache["vlm_sent_count"] == 0:
        if cached_crop_count < cache_size:
            return None
        dispatch_mode = "initial_candidate"
        dispatch_reason = "cache_full"
    elif track_cache["vlm_retry_requested"]:
        if track_cache["last_status"] not in {"new", "active"}:
            return None
        if cached_crop_count < cache_size:
            return None
        dispatch_mode = "retry_candidate"
        dispatch_reason = "retry_requested_refill_complete"
    else:
        return None

    dispatched_package = {
        "vlm_frame_cropper_layer_track_id": selected_crop["track_id"],
        "vlm_frame_cropper_layer_image": selected_crop["crop"].copy(),
        "vlm_frame_cropper_layer_bbox": tuple(selected_crop["bbox"]),
    }
    track_cache["vlm_request_in_flight"] = True
    track_cache["vlm_retry_requested"] = False
    track_cache["vlm_previous_sent_must_be_used"] = False
    track_cache["vlm_sent_count"] += 1
    if dispatch_mode == "dead_best_available":
        track_cache["vlm_terminal_state"] = "dead"
    track_cache["vlm_last_sent_frame_id"] = selected_crop["frame_id"]
    track_cache["vlm_last_sent_selection_key"] = selected_crop["selection_key"]
    track_cache["vlm_last_dispatch_reason"] = dispatch_reason
    track_cache["vlm_last_dispatch_mode"] = dispatch_mode
    track_cache["vlm_last_sent_package"] = {
        "frame_id": selected_crop["frame_id"],
        "track_id": selected_crop["track_id"],
        "bbox": tuple(selected_crop["bbox"]),
        "trigger_reason": selected_crop["trigger_reason"],
        "confidence": selected_crop["confidence"],
        "crop": selected_crop["crop"].copy(),
    }

    return {
        "vlm_dispatch_track_id": track_cache["track_id"],
        "vlm_dispatch_mode": dispatch_mode,
        "vlm_dispatch_reason": dispatch_reason,
        "vlm_dispatch_cached_crop_count": cached_crop_count,
        "vlm_frame_cropper_layer_package": dispatched_package,
    }


def register_vlm_ack_package(vlm_crop_cache_state: dict[str, Any], vlm_ack_package: Any) -> dict[str, Any]:
    _validate_vlm_crop_cache_state(vlm_crop_cache_state)
    _validate_vlm_ack_package(vlm_ack_package)
    track_cache = _get_track_cache(vlm_crop_cache_state, _package_get(vlm_ack_package, "vlm_ack_track_id"))
    if track_cache is None:
        raise KeyError("vlm_ack_track_id was not found in vlm_crop_cache_state.")

    ack_status = str(_package_get(vlm_ack_package, "vlm_ack_status"))
    retry_requested = bool(_package_get(vlm_ack_package, "vlm_ack_retry_requested"))
    ack_reason = str(_package_get(vlm_ack_package, "vlm_ack_reason"))

    track_cache["vlm_request_in_flight"] = False
    track_cache["vlm_ack_status"] = ack_status
    track_cache["vlm_ack_reason"] = ack_reason
    if ack_status == "accepted":
        track_cache["vlm_retry_requested"] = False
        track_cache["vlm_finalized"] = True
        track_cache["vlm_previous_sent_must_be_used"] = False
        if ack_reason == _NO_ACK_REASON:
            track_cache["vlm_terminal_state"] = "no"
        else:
            track_cache["vlm_terminal_state"] = "done"
    elif ack_status == "retry_requested":
        track_cache["vlm_retry_requested"] = retry_requested or True
        track_cache["vlm_previous_sent_must_be_used"] = False
        track_cache["cached_crops"] = []
        track_cache["selected_crop"] = None
        track_cache["vlm_terminal_state"] = "collecting"
    elif ack_status == "finalize_with_current":
        track_cache["vlm_retry_requested"] = False
        track_cache["vlm_finalized"] = True
        track_cache["vlm_previous_sent_must_be_used"] = True
        track_cache["vlm_terminal_state"] = "done"
    return deepcopy(track_cache)


def select_best_vlm_crop_candidate(cached_crops: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not cached_crops:
        return None
    return max(cached_crops, key=lambda crop: crop["selection_key"])


def score_vlm_crop_candidate(cached_crop: dict[str, Any]) -> float:
    bbox = cached_crop["bbox"]
    area = max(1, (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1])))
    confidence_term = float(cached_crop["confidence"])
    area_term = min(1.0, area / CROP_SELECTION_AREA_NORMALIZER)
    recency_term = min(1.0, int(cached_crop["frame_id"]) / CROP_SELECTION_FRAME_NORMALIZER)
    return (CROP_SELECTION_CONFIDENCE_WEIGHT * confidence_term) + (CROP_SELECTION_AREA_WEIGHT * area_term) + (CROP_SELECTION_RECENCY_WEIGHT * recency_term)


def resolve_source_frame(input_layer_package: Any, vlm_frame_cropper_request_package: dict[str, Any]) -> np.ndarray:
    _validate_input_layer_package(input_layer_package)
    _validate_request_package(vlm_frame_cropper_request_package)
    if _package_get(input_layer_package, "input_layer_frame_id") != vlm_frame_cropper_request_package["vlm_frame_cropper_frame_id"]:
        raise ValueError("input_layer_frame_id does not match vlm_frame_cropper_frame_id.")
    return _package_get(input_layer_package, "input_layer_image")


def crop_object_from_frame(source_frame: np.ndarray, vlm_frame_cropper_bbox: tuple[int | float, int | float, int | float, int | float]) -> np.ndarray:
    expanded_bbox = _expand_bbox_with_context(
        bbox=vlm_frame_cropper_bbox,
        frame_image=source_frame,
        padding_ratio=VLM_CROP_CONTEXT_PADDING_RATIO,
    )
    x_min, y_min, x_max, y_max = _normalize_bbox(bbox=expanded_bbox, frame_image=source_frame)
    return source_frame[y_min:y_max, x_min:x_max].copy()


def validate_crop_result(vlm_object_crop: np.ndarray) -> None:
    if not isinstance(vlm_object_crop, np.ndarray):
        raise TypeError("vlm_object_crop must be a numpy ndarray.")
    if vlm_object_crop.size == 0:
        raise ValueError("vlm_object_crop is empty.")
    if len(vlm_object_crop.shape) < 2:
        raise ValueError("vlm_object_crop must have at least 2 dimensions.")
    if vlm_object_crop.shape[0] <= 0 or vlm_object_crop.shape[1] <= 0:
        raise ValueError("vlm_object_crop has invalid spatial dimensions.")


def _validate_input_layer_package(input_layer_package: Any) -> None:
    _validate_required_fields(package=input_layer_package, required_fields=["input_layer_frame_id", "input_layer_timestamp", "input_layer_image", "input_layer_source_type", "input_layer_resolution"], package_name="input_layer_package")


def _validate_tracking_layer_package(tracking_layer_package: dict[str, Any]) -> None:
    required_fields = ["tracking_layer_frame_id", "tracking_layer_track_id", "tracking_layer_bbox", "tracking_layer_detector_class", "tracking_layer_confidence", "tracking_layer_status"]
    missing_fields = [field_name for field_name in required_fields if field_name not in tracking_layer_package]
    if missing_fields:
        raise ValueError("tracking_layer_package is missing required fields: " + ", ".join(missing_fields))
    list_lengths = {field_name: len(tracking_layer_package[field_name]) for field_name in required_fields if field_name != "tracking_layer_frame_id"}
    if len(set(list_lengths.values())) > 1:
        raise ValueError("tracking_layer_package list fields must all have the same length.")


def _validate_request_package(vlm_frame_cropper_request_package: dict[str, Any]) -> None:
    required_fields = ["vlm_frame_cropper_frame_id", "vlm_frame_cropper_track_id", "vlm_frame_cropper_bbox", "vlm_frame_cropper_trigger_reason"]
    missing_fields = [field_name for field_name in required_fields if field_name not in vlm_frame_cropper_request_package]
    if missing_fields:
        raise ValueError("vlm_frame_cropper_request_package is missing required fields: " + ", ".join(missing_fields))


def _validate_cropper_layer_package(vlm_frame_cropper_layer_package: dict[str, Any]) -> None:
    required_fields = ["vlm_frame_cropper_layer_track_id", "vlm_frame_cropper_layer_image", "vlm_frame_cropper_layer_bbox"]
    missing_fields = [field_name for field_name in required_fields if field_name not in vlm_frame_cropper_layer_package]
    if missing_fields:
        raise ValueError("vlm_frame_cropper_layer_package is missing required fields: " + ", ".join(missing_fields))
    validate_crop_result(vlm_frame_cropper_layer_package["vlm_frame_cropper_layer_image"])


def _validate_vlm_crop_cache_state(vlm_crop_cache_state: dict[str, Any]) -> None:
    required = {"config_vlm_crop_cache_size", "config_vlm_dead_after_lost_frames", "track_caches"}
    if not required.issubset(vlm_crop_cache_state):
        raise ValueError("vlm_crop_cache_state is missing required fields.")


def _validate_tracking_layer_row(tracking_layer_row: dict[str, Any]) -> None:
    required_fields = ["track_id", "bbox", "detector_class", "confidence", "status"]
    missing_fields = [field_name for field_name in required_fields if field_name not in tracking_layer_row]
    if missing_fields:
        raise ValueError("tracking_layer_row is missing required fields: " + ", ".join(missing_fields))


def _validate_vlm_ack_package(vlm_ack_package: Any) -> None:
    _validate_required_fields(package=vlm_ack_package, required_fields=["vlm_ack_track_id", "vlm_ack_status", "vlm_ack_reason", "vlm_ack_retry_requested"], package_name="vlm_ack_package")
    ack_status = str(_package_get(vlm_ack_package, "vlm_ack_status"))
    if ack_status not in _ALLOWED_VLM_ACK_STATUSES:
        raise ValueError("vlm_ack_status must be one of: accepted, retry_requested, finalize_with_current.")


def _ensure_track_cache_entry(vlm_crop_cache_state: dict[str, Any], track_id: Any) -> dict[str, Any]:
    track_key = str(track_id)
    if track_key not in vlm_crop_cache_state["track_caches"]:
        vlm_crop_cache_state["track_caches"][track_key] = {
            "track_id": track_key,
            "cached_crops": [],
            "selected_crop": None,
            "last_status": "unknown",
            "last_frame_id": -1,
            "detector_class": "unknown",
            "vlm_sent_count": 0,
            "lost_frame_streak": 0,
            "vlm_request_in_flight": False,
            "vlm_retry_requested": False,
            "vlm_ack_status": "not_requested",
            "vlm_ack_reason": "",
            "vlm_last_sent_frame_id": None,
            "vlm_last_sent_selection_key": None,
            "vlm_last_dispatch_reason": "",
            "vlm_last_dispatch_mode": "",
            "vlm_last_sent_package": None,
            "vlm_previous_sent_must_be_used": False,
            "vlm_finalized": False,
            "vlm_dead": False,
            "vlm_terminal_state": "collecting",
        }
    return vlm_crop_cache_state["track_caches"][track_key]


def _get_track_cache(vlm_crop_cache_state: dict[str, Any], track_id: Any) -> dict[str, Any] | None:
    return vlm_crop_cache_state["track_caches"].get(str(track_id))


def _validate_required_fields(package: Any, required_fields: list[str], package_name: str) -> None:
    missing_fields = [field_name for field_name in required_fields if not _package_has(package, field_name)]
    if missing_fields:
        raise ValueError(f"{package_name} is missing required fields: {', '.join(missing_fields)}")


def _package_has(package: Any, field_name: str) -> bool:
    return (isinstance(package, dict) and field_name in package) or hasattr(package, field_name)


def _package_get(package: Any, field_name: str) -> Any:
    if isinstance(package, dict):
        return package[field_name]
    if is_dataclass(package):
        return asdict(package)[field_name]
    return getattr(package, field_name)


def _normalize_bbox(bbox: tuple[int | float, int | float, int | float, int | float], frame_image: np.ndarray) -> tuple[int, int, int, int]:
    frame_height, frame_width = frame_image.shape[:2]
    x_min = max(0, min(int(np.floor(bbox[0])), frame_width - 1))
    y_min = max(0, min(int(np.floor(bbox[1])), frame_height - 1))
    x_max = max(x_min + 1, min(int(np.ceil(bbox[2])), frame_width))
    y_max = max(y_min + 1, min(int(np.ceil(bbox[3])), frame_height))
    return (x_min, y_min, x_max, y_max)


def _expand_bbox_with_context(
    bbox: tuple[int | float, int | float, int | float, int | float],
    frame_image: np.ndarray,
    padding_ratio: float,
) -> tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = bbox
    bbox_width = max(1.0, float(x_max) - float(x_min))
    bbox_height = max(1.0, float(y_max) - float(y_min))

    pad_x = bbox_width * float(padding_ratio)
    pad_y = bbox_height * float(padding_ratio)

    return (
        float(x_min) - pad_x,
        float(y_min) - pad_y,
        float(x_max) + pad_x,
        float(y_max) + pad_y,
    )


__all__ = [
    "build_vlm_dispatch_package",
    "build_vlm_frame_cropper_package",
    "build_vlm_frame_cropper_request_package",
    "crop_object_from_frame",
    "extract_vlm_object_crop",
    "initialize_vlm_crop_cache",
    "refresh_vlm_crop_cache_track_state",
    "register_vlm_ack_package",
    "resolve_source_frame",
    "score_vlm_crop_candidate",
    "CROP_SELECTION_CONFIDENCE_WEIGHT",
    "CROP_SELECTION_AREA_WEIGHT",
    "CROP_SELECTION_RECENCY_WEIGHT",
    "CROP_SELECTION_AREA_NORMALIZER",
    "CROP_SELECTION_FRAME_NORMALIZER",
    "VLM_CROP_CONTEXT_PADDING_RATIO",
    "select_best_vlm_crop_candidate",
    "update_vlm_crop_cache",
    "validate_crop_result",
]
