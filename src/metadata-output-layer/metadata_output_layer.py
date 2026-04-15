from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable


def build_metadata_output_layer_package(
    vehicle_state_layer_package: dict,
    vlm_layer_package: dict | None = None,
    scene_awareness_layer_package: dict | None = None,
) -> dict:
    """
    Assemble the structured metadata package for downstream emission.

    Contract source: `pipeline/pipeline_layers_and_interactions.md` (Layer 9).

    Expected upstream shape (v1): dict of parallel lists in `vehicle_state_layer_package`.
    This function is defensive and will degrade gracefully if optional packages are missing.
    """
    now_iso = _iso_utc_now()

    vehicle_rows = _rows_from_parallel_lists(
        vehicle_state_layer_package,
        required_key="vehicle_state_layer_track_id",
    )

    # Build optional enrichments keyed by track_id.
    vlm_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(vlm_layer_package, dict) and vlm_layer_package:
        vlm_rows = _rows_from_parallel_lists(vlm_layer_package, required_key="vlm_layer_track_id")
        for r in vlm_rows:
            track_id = _stringify_id(r.get("vlm_layer_track_id"))
            if track_id:
                vlm_by_id[track_id] = r

    scene_tags: list[str] = _scene_tags_from_package(scene_awareness_layer_package)

    # Deterministic ordering: sort by track_id (numeric if possible).
    vehicle_rows_sorted = sorted(vehicle_rows, key=lambda r: _sort_key_for_id(r.get("vehicle_state_layer_track_id")))

    object_ids: list[str] = []
    classes: list[str | None] = []
    semantic_tags: list[list[str]] = []
    timestamps: list[str] = []

    # Optional per-object summaries as structured dicts (kept stable and minimal).
    per_object_summaries: list[dict[str, Any]] = []

    for r in vehicle_rows_sorted:
        track_id = _stringify_id(r.get("vehicle_state_layer_track_id"))
        if not track_id:
            # Skip malformed rows; keep deterministic behavior (no random IDs).
            continue

        obj_class = _none_if_empty(r.get("vehicle_state_layer_vehicle_class"))
        tags = _normalize_tags(r.get("vehicle_state_layer_semantic_tags"))

        # Merge in VLM info (non-persistent; do not overwrite stored tags unless additive).
        if track_id in vlm_by_id:
            vlm_r = vlm_by_id[track_id]
            vlm_label = _none_if_empty(vlm_r.get("vlm_layer_label"))
            vlm_attrs = _normalize_tags(vlm_r.get("vlm_layer_attributes"))
            if vlm_label and vlm_label not in tags:
                tags.append(vlm_label)
            for t in vlm_attrs:
                if t not in tags:
                    tags.append(t)

        object_ids.append(track_id)
        classes.append(obj_class)
        semantic_tags.append(tags)
        timestamps.append(now_iso)

        per_object_summaries.append(
            {
                "object_id": track_id,
                "class": obj_class,
                "semantic_tag_count": len(tags),
                "first_seen_frame": r.get("vehicle_state_layer_first_seen_frame"),
                "last_seen_frame": r.get("vehicle_state_layer_last_seen_frame"),
                "lost_frame_count": r.get("vehicle_state_layer_lost_frame_count"),
                "vlm_called": r.get("vehicle_state_layer_vlm_called"),
                "truck_type": _none_if_empty(r.get("vehicle_state_layer_truck_type")),
            }
        )

    counts = _build_counts(object_ids=object_ids, classes=classes, semantic_tags=semantic_tags, scene_tags=scene_tags)

    summaries = {
        "generated_at": now_iso,
        "object_summary": per_object_summaries,
        "high_level": {
            "total_objects": counts["total_objects"],
            "top_classes": counts["classes_by_count"][:5],
            "top_semantic_tags": counts["semantic_tags_by_count"][:10],
            "scene_tags_present": scene_tags,
        },
    }

    return {
        "metadata_output_layer_timestamps": timestamps,
        "metadata_output_layer_object_ids": object_ids,
        "metadata_output_layer_classes": classes,
        "metadata_output_layer_semantic_tags": semantic_tags,
        "metadata_output_layer_scene_tags": scene_tags,
        "metadata_output_layer_counts": counts,
        "metadata_output_layer_summaries": summaries,
    }


def serialize_metadata_output(metadata_output_layer_package: dict, output_format: str = "json") -> str | bytes:
    """
    Convert metadata into the selected output format.

    Supported formats:
    - "json": returns a deterministic JSON string.
    - "json_bytes": returns UTF-8 encoded bytes of the same deterministic JSON.
    """
    if output_format not in {"json", "json_bytes"}:
        raise ValueError(f"Unsupported output_format: {output_format!r}")

    payload_str = json.dumps(
        metadata_output_layer_package,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    if output_format == "json_bytes":
        return payload_str.encode("utf-8")
    return payload_str


def emit_metadata_output(
    serialized_payload: str | bytes,
    output_destination: str = "stdout",
    output_path: str | None = None,
) -> None:
    """
    Send metadata to the configured output destination.

    Destinations:
    - "stdout": write to standard output
    - "file": write to `output_path` (required)
    """
    if output_destination not in {"stdout", "file"}:
        raise ValueError(f"Unsupported output_destination: {output_destination!r}")

    if output_destination == "stdout":
        if isinstance(serialized_payload, bytes):
            sys.stdout.buffer.write(serialized_payload)
            sys.stdout.buffer.write(b"\n")
        else:
            sys.stdout.write(serialized_payload)
            sys.stdout.write("\n")
        return

    # file
    if not output_path:
        raise ValueError("output_path is required when output_destination='file'")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    if isinstance(serialized_payload, bytes):
        with open(output_path, "wb") as f:
            f.write(serialized_payload)
            f.write(b"\n")
    else:
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(serialized_payload)
            f.write("\n")


# -------------------------
# Internal helpers (stdlib)
# -------------------------


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stringify_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and value != value:  # NaN
        return None
    s = str(value).strip()
    return s or None


def _sort_key_for_id(value: Any) -> tuple[int, Any]:
    """
    Deterministic sort key for object IDs:
    - numeric IDs first in numeric order
    - then non-numeric IDs in lexicographic order
    """
    s = _stringify_id(value)
    if s is None:
        return (2, "")
    try:
        return (0, int(s))
    except Exception:
        return (1, s)


def _none_if_empty(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v or None
    return str(value)


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    if isinstance(value, dict):
        # Convert dict attributes into stable "k=v" tags sorted by key.
        tags: list[str] = []
        for k in sorted(value.keys(), key=lambda x: str(x)):
            tags.append(f"{k}={value[k]}")
        return tags
    if isinstance(value, (list, tuple, set)):
        tags = [_none_if_empty(v) for v in value]
        return sorted([t for t in tags if t], key=lambda x: x)
    # Fallback to string tag.
    s = _none_if_empty(value)
    return [s] if s else []


def _rows_from_parallel_lists(package: dict, required_key: str) -> list[dict[str, Any]]:
    """
    Convert a dict-of-parallel-lists package into a list of row dicts.
    If the required key is missing or not list-like, returns [].
    """
    if not isinstance(package, dict) or required_key not in package:
        return []

    required_vals = package.get(required_key)
    if not isinstance(required_vals, (list, tuple)):
        return []

    # Identify keys that are parallel sequences of the same length.
    n = len(required_vals)
    parallel_keys: list[str] = []
    for k, v in package.items():
        if isinstance(v, (list, tuple)) and len(v) == n:
            parallel_keys.append(k)

    parallel_keys_sorted = sorted(parallel_keys)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row = {k: package[k][i] for k in parallel_keys_sorted}
        rows.append(row)
    return rows


def _scene_tags_from_package(scene_awareness_layer_package: dict | None) -> list[str]:
    if not isinstance(scene_awareness_layer_package, dict) or not scene_awareness_layer_package:
        return []

    label = _none_if_empty(scene_awareness_layer_package.get("scene_awareness_layer_label"))
    attrs = _normalize_tags(scene_awareness_layer_package.get("scene_awareness_layer_attributes"))
    tags: list[str] = []
    if label:
        tags.append(label)
    tags.extend(attrs)
    return sorted(set(tags), key=lambda x: x)


def _build_counts(
    object_ids: list[str],
    classes: list[str | None],
    semantic_tags: list[list[str]],
    scene_tags: list[str],
) -> dict[str, Any]:
    class_counter = Counter([c for c in classes if c])
    tag_counter = Counter([t for tags in semantic_tags for t in tags if t])
    scene_counter = Counter(scene_tags)

    return {
        "total_objects": len(object_ids),
        "classes_by_count": _counter_to_sorted_list(class_counter),
        "semantic_tags_by_count": _counter_to_sorted_list(tag_counter),
        "scene_tags_by_count": _counter_to_sorted_list(scene_counter),
    }


def _counter_to_sorted_list(counter: Counter[str]) -> list[dict[str, Any]]:
    # Deterministic: sort by count desc then key asc.
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"value": k, "count": v} for k, v in items]


def _json_default(obj: Any) -> Any:
    """
    Keep serialization stable and safe for a few common non-JSON stdlib types.
    """
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=timezone.utc)
        return obj.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if isinstance(obj, set):
        return sorted(obj)
    # Last resort: stable string representation.
    return str(obj)

