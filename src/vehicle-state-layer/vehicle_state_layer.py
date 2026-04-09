"""
vehicle_state_layer.py

Layer 6: Vehicle State

Purpose:
    Store persistent per-vehicle metadata across frames.

Public functions (from pipeline_layers_and_interactions.md):
    initialize_vehicle_state_layer
    update_vehicle_state_from_tracking
    update_vehicle_state_from_vlm
    get_vehicle_state_record
    build_vehicle_state_layer_package

This layer owns persistent metadata only. It does not decide whether a track is
new, active, or lost; that decision belongs to the tracking layer.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_DEFAULT_TRUCK_TYPE = "unknown"


_state = {
    "initialized": False,
    "records": {},
    "prune_after_lost_frames": None,
}


def initialize_vehicle_state_layer(prune_after_lost_frames: int | None = None) -> None:
    """
    Prepare empty persistent vehicle state.

    Args:
        prune_after_lost_frames:
            Optional threshold for dropping stale records once their
            ``vehicle_state_layer_lost_frame_count`` exceeds this value.
            ``None`` keeps lost records indefinitely.
    """
    if prune_after_lost_frames is not None and prune_after_lost_frames < 0:
        raise ValueError("prune_after_lost_frames must be non-negative or None.")

    _state["initialized"] = True
    _state["records"] = {}
    _state["prune_after_lost_frames"] = prune_after_lost_frames


def update_vehicle_state_from_tracking(
    tracking_layer_package: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """
    Update object state using tracking data.

    Expected tracking package fields:
        - tracking_layer_frame_id
        - tracking_layer_track_id
        - tracking_layer_bbox
        - tracking_layer_detector_class
        - tracking_layer_confidence
        - tracking_layer_status

    Returns:
        A deep-copied mapping of ``track_id -> vehicle_state_record`` after the
        tracking update and optional pruning pass.
    """
    _assert_initialized()
    _validate_tracking_layer_package(tracking_layer_package)

    frame_id = tracking_layer_package["tracking_layer_frame_id"]
    track_ids = tracking_layer_package["tracking_layer_track_id"]
    detector_classes = tracking_layer_package["tracking_layer_detector_class"]
    statuses = tracking_layer_package["tracking_layer_status"]

    for index, track_id in enumerate(track_ids):
        track_id_str = str(track_id)
        record = _state["records"].get(track_id_str)
        if record is None:
            record = create_vehicle_state_record(
                vehicle_state_layer_track_id=track_id_str,
                vehicle_state_layer_first_seen_frame=frame_id,
                vehicle_state_layer_vehicle_class=detector_classes[index],
            )
            _state["records"][track_id_str] = record

        merge_tracking_into_vehicle_state(
            vehicle_state_record=record,
            tracking_frame_id=frame_id,
            tracking_detector_class=detector_classes[index],
            tracking_status=statuses[index],
        )

    prune_vehicle_state_records()
    return deepcopy(_state["records"])


def update_vehicle_state_from_vlm(
    vlm_layer_package: dict[str, Any],
) -> dict[str, Any]:
    """
    Update stored semantic fields using VLM results.

    Expected VLM package fields:
        - vlm_layer_track_id
        - vlm_layer_label
        - vlm_layer_attributes

    Returns:
        The updated record for the target track as a deep copy.
    """
    _assert_initialized()
    _validate_vlm_layer_package(vlm_layer_package)

    track_id = str(vlm_layer_package["vlm_layer_track_id"])
    record = _state["records"].get(track_id)
    if record is None:
        raise KeyError(
            "vlm_layer_track_id was not found in vehicle_state_layer. "
            "Tracking data must create the record before VLM enrichment."
        )

    merge_vlm_into_vehicle_state(
        vehicle_state_record=record,
        vlm_layer_label=vlm_layer_package["vlm_layer_label"],
        vlm_layer_attributes=vlm_layer_package["vlm_layer_attributes"],
    )
    return deepcopy(record)


def get_vehicle_state_record(vehicle_state_layer_track_id: str | int) -> dict[str, Any] | None:
    """Return the current state record for one track, or ``None`` if missing."""
    _assert_initialized()
    record = _state["records"].get(str(vehicle_state_layer_track_id))
    return deepcopy(record) if record is not None else None


def build_vehicle_state_layer_package(
    track_ids: list[str | int] | None = None,
) -> dict[str, list[Any]]:
    """
    Create the vehicle_state_layer_package for downstream output.

    Args:
        track_ids:
            Optional list of track IDs to include. When omitted, all records are
            emitted in ascending numeric-then-lexicographic track ID order.
    """
    _assert_initialized()

    selected_records = _select_records_for_package(track_ids=track_ids)

    return {
        "vehicle_state_layer_track_id": [
            record["vehicle_state_layer_track_id"] for record in selected_records
        ],
        "vehicle_state_layer_first_seen_frame": [
            record["vehicle_state_layer_first_seen_frame"] for record in selected_records
        ],
        "vehicle_state_layer_last_seen_frame": [
            record["vehicle_state_layer_last_seen_frame"] for record in selected_records
        ],
        "vehicle_state_layer_lost_frame_count": [
            record["vehicle_state_layer_lost_frame_count"] for record in selected_records
        ],
        "vehicle_state_layer_vehicle_class": [
            record["vehicle_state_layer_vehicle_class"] for record in selected_records
        ],
        "vehicle_state_layer_truck_type": [
            record["vehicle_state_layer_truck_type"] for record in selected_records
        ],
        "vehicle_state_layer_semantic_tags": [
            deepcopy(record["vehicle_state_layer_semantic_tags"])
            for record in selected_records
        ],
        "vehicle_state_layer_vlm_called": [
            record["vehicle_state_layer_vlm_called"] for record in selected_records
        ],
    }


def create_vehicle_state_record(
    vehicle_state_layer_track_id: str,
    vehicle_state_layer_first_seen_frame: int,
    vehicle_state_layer_vehicle_class: str,
) -> dict[str, Any]:
    """Create the initial state record for a newly tracked object."""
    return {
        "vehicle_state_layer_track_id": vehicle_state_layer_track_id,
        "vehicle_state_layer_first_seen_frame": vehicle_state_layer_first_seen_frame,
        "vehicle_state_layer_last_seen_frame": vehicle_state_layer_first_seen_frame,
        "vehicle_state_layer_lost_frame_count": 0,
        "vehicle_state_layer_vehicle_class": vehicle_state_layer_vehicle_class,
        "vehicle_state_layer_truck_type": _DEFAULT_TRUCK_TYPE,
        "vehicle_state_layer_semantic_tags": {},
        "vehicle_state_layer_vlm_called": False,
    }


def merge_tracking_into_vehicle_state(
    vehicle_state_record: dict[str, Any],
    tracking_frame_id: int,
    tracking_detector_class: str,
    tracking_status: str,
) -> None:
    """Apply tracking updates to the stored state record."""
    if tracking_status not in {"new", "active", "lost"}:
        raise ValueError(
            "tracking_status must be one of: 'new', 'active', 'lost'."
        )

    vehicle_state_record["vehicle_state_layer_vehicle_class"] = tracking_detector_class

    if tracking_status in {"new", "active"}:
        vehicle_state_record["vehicle_state_layer_last_seen_frame"] = tracking_frame_id
        vehicle_state_record["vehicle_state_layer_lost_frame_count"] = 0
        return

    vehicle_state_record["vehicle_state_layer_lost_frame_count"] += 1


def merge_vlm_into_vehicle_state(
    vehicle_state_record: dict[str, Any],
    vlm_layer_label: str,
    vlm_layer_attributes: dict[str, Any],
) -> None:
    """Apply semantic enrichment to the stored state record."""
    semantic_tags = vehicle_state_record["vehicle_state_layer_semantic_tags"]
    semantic_tags.update(deepcopy(vlm_layer_attributes))

    vehicle_state_record["vehicle_state_layer_truck_type"] = (
        vlm_layer_attributes.get("truck_type")
        or vlm_layer_label
        or _DEFAULT_TRUCK_TYPE
    )
    vehicle_state_record["vehicle_state_layer_vlm_called"] = True


def prune_vehicle_state_records() -> None:
    """Remove stale records when a prune threshold is configured."""
    threshold = _state["prune_after_lost_frames"]
    if threshold is None:
        return

    stale_track_ids = [
        track_id
        for track_id, record in _state["records"].items()
        if record["vehicle_state_layer_lost_frame_count"] > threshold
    ]
    for track_id in stale_track_ids:
        del _state["records"][track_id]


def _assert_initialized() -> None:
    if not _state["initialized"]:
        raise RuntimeError(
            "vehicle_state_layer has not been initialized. "
            "Call initialize_vehicle_state_layer() first."
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
    _validate_required_fields(
        package=tracking_layer_package,
        required_fields=required_fields,
        package_name="tracking_layer_package",
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


def _validate_vlm_layer_package(vlm_layer_package: dict[str, Any]) -> None:
    required_fields = [
        "vlm_layer_track_id",
        "vlm_layer_label",
        "vlm_layer_attributes",
    ]
    _validate_required_fields(
        package=vlm_layer_package,
        required_fields=required_fields,
        package_name="vlm_layer_package",
    )

    if not isinstance(vlm_layer_package["vlm_layer_attributes"], dict):
        raise ValueError("vlm_layer_attributes must be a dict.")


def _validate_required_fields(
    package: dict[str, Any],
    required_fields: list[str],
    package_name: str,
) -> None:
    missing_fields = [
        field_name for field_name in required_fields if field_name not in package
    ]
    if missing_fields:
        raise ValueError(
            f"{package_name} is missing required fields: {', '.join(missing_fields)}"
        )


def _select_records_for_package(track_ids: list[str | int] | None) -> list[dict[str, Any]]:
    if track_ids is None:
        selected_track_ids = sorted(_state["records"].keys(), key=_track_id_sort_key)
    else:
        selected_track_ids = [str(track_id) for track_id in track_ids]

    selected_records = []
    for track_id in selected_track_ids:
        record = _state["records"].get(track_id)
        if record is not None:
            selected_records.append(record)
    return selected_records


def _track_id_sort_key(track_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(track_id))
    except ValueError:
        return (1, track_id)


__all__ = [
    "build_vehicle_state_layer_package",
    "create_vehicle_state_record",
    "get_vehicle_state_record",
    "initialize_vehicle_state_layer",
    "merge_tracking_into_vehicle_state",
    "merge_vlm_into_vehicle_state",
    "prune_vehicle_state_records",
    "update_vehicle_state_from_tracking",
    "update_vehicle_state_from_vlm",
]

