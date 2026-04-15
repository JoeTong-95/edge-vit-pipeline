from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any


def _safe_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 1


def _pluck(package: Any, key: str, default: Any = None) -> Any:
    if package is None:
        return default
    if isinstance(package, dict):
        return package.get(key, default)
    return getattr(package, key, default)


def _count_detections(yolo_layer_package: Any) -> int:
    detections = _pluck(yolo_layer_package, "yolo_layer_detections", None)
    return _safe_len(detections)


def _count_tracks(tracking_layer_package: Any) -> int:
    # Contract describes per-track fields; in practice this may be a list of records.
    if tracking_layer_package is None:
        return 0
    if isinstance(tracking_layer_package, list):
        return len(tracking_layer_package)
    tracks = _pluck(tracking_layer_package, "tracking_layer_tracks", None)
    if tracks is not None:
        return _safe_len(tracks)
    return 1


def _count_vlm_calls(vlm_layer_package: Any) -> int:
    if vlm_layer_package is None:
        return 0
    if isinstance(vlm_layer_package, list):
        return len(vlm_layer_package)
    calls = _pluck(vlm_layer_package, "vlm_layer_calls", None)
    if calls is not None:
        return _safe_len(calls)
    return 1


def _count_scene_calls(scene_awareness_layer_package: Any) -> int:
    if scene_awareness_layer_package is None:
        return 0
    if isinstance(scene_awareness_layer_package, list):
        return len(scene_awareness_layer_package)
    calls = _pluck(scene_awareness_layer_package, "scene_awareness_layer_calls", None)
    if calls is not None:
        return _safe_len(calls)
    return 1


def _estimate_fps(timings: dict | None) -> float | None:
    if not timings:
        return None

    for key in ("pipeline_total_s", "end_to_end_s", "frame_duration_s", "total_s"):
        v = timings.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return 1.0 / float(v)

    durations = timings.get("frame_durations_s")
    if isinstance(durations, (list, tuple)) and durations:
        vals = [float(x) for x in durations if isinstance(x, (int, float)) and x > 0]
        if vals:
            return 1.0 / (sum(vals) / len(vals))

    start = timings.get("frame_start_s")
    end = timings.get("frame_end_s")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
        return 1.0 / float(end - start)

    return None


def _extract_module_latency(timings: dict | None) -> dict:
    if not timings:
        return {}

    module_latency = timings.get("module_latency")
    if isinstance(module_latency, dict):
        return dict(module_latency)

    out: dict[str, float] = {}
    for k, v in timings.items():
        if not isinstance(v, (int, float)):
            continue
        if k.endswith("_latency_s"):
            out[k.removesuffix("_latency_s")] = float(v)
        elif k.endswith("_latency_ms"):
            out[k.removesuffix("_latency_ms")] = float(v) / 1000.0
    return out


def collect_evaluation_metrics(
    input_layer_package=None,
    roi_layer_package=None,
    yolo_layer_package=None,
    tracking_layer_package=None,
    vlm_layer_package=None,
    scene_awareness_layer_package=None,
    timings: dict | None = None,
) -> dict:
    """
    Gather metrics from active layers (Layer 10 contract).
    """
    metrics: dict[str, Any] = {}

    metrics["fps"] = _estimate_fps(timings)
    metrics["module_latency"] = _extract_module_latency(timings)

    metrics["detection_count"] = _count_detections(yolo_layer_package)
    metrics["track_count"] = _count_tracks(tracking_layer_package)
    metrics["vlm_call_count"] = _count_vlm_calls(vlm_layer_package)
    metrics["scene_call_count"] = _count_scene_calls(scene_awareness_layer_package)

    # Optional identifiers for debugging/joins (not required by contract).
    metrics["frame_id"] = (
        _pluck(input_layer_package, "input_layer_frame_id", None)
        or _pluck(roi_layer_package, "roi_layer_frame_id", None)
        or _pluck(yolo_layer_package, "yolo_layer_frame_id", None)
        or _pluck(tracking_layer_package, "tracking_layer_frame_id", None)
        or _pluck(scene_awareness_layer_package, "scene_awareness_layer_frame_id", None)
    )
    metrics["timestamp"] = (
        _pluck(input_layer_package, "input_layer_timestamp", None)
        or _pluck(roi_layer_package, "roi_layer_timestamp", None)
        or _pluck(scene_awareness_layer_package, "scene_awareness_layer_timestamp", None)
        or time.time()
    )

    return metrics


def build_evaluation_output_layer_package(metrics: dict) -> dict:
    """
    Create the `evaluation_output_layer_package` per the pipeline contract.
    """
    return {
        "evaluation_output_layer_fps": metrics.get("fps"),
        "evaluation_output_layer_module_latency": metrics.get("module_latency", {}) or {},
        "evaluation_output_layer_detection_count": int(metrics.get("detection_count", 0) or 0),
        "evaluation_output_layer_track_count": int(metrics.get("track_count", 0) or 0),
        "evaluation_output_layer_vlm_call_count": int(metrics.get("vlm_call_count", 0) or 0),
        "evaluation_output_layer_scene_call_count": int(metrics.get("scene_call_count", 0) or 0),
        # Optional debug fields.
        "evaluation_output_layer_frame_id": metrics.get("frame_id"),
        "evaluation_output_layer_timestamp": metrics.get("timestamp"),
    }


def emit_evaluation_output(
    evaluation_output_layer_package,
    output_destination: str = "sqlite",
    output_path: str = "evaluation.sqlite",
) -> None:
    """
    Write evaluation records to the configured sink.

    Minimum viable sink: sqlite3 database file at `output_path`.
    """
    if output_destination not in {"sqlite", "stdout"}:
        raise ValueError(f"Unsupported output_destination: {output_destination!r}")

    record = evaluation_output_layer_package
    if not isinstance(record, dict):
        raise TypeError("evaluation_output_layer_package must be a dict")

    if output_destination == "stdout":
        print(json.dumps(record, separators=(",", ":"), ensure_ascii=False, sort_keys=True))
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    module_latency_json = json.dumps(
        record.get("evaluation_output_layer_module_latency", {}) or {},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    record_json = json.dumps(record, separators=(",", ":"), ensure_ascii=False)

    conn = sqlite3.connect(output_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_records (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at_s REAL NOT NULL,
              frame_id TEXT,
              fps REAL,
              detection_count INTEGER NOT NULL,
              track_count INTEGER NOT NULL,
              vlm_call_count INTEGER NOT NULL,
              scene_call_count INTEGER NOT NULL,
              module_latency_json TEXT NOT NULL,
              evaluation_package_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO evaluation_records (
              created_at_s,
              frame_id,
              fps,
              detection_count,
              track_count,
              vlm_call_count,
              scene_call_count,
              module_latency_json,
              evaluation_package_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                float(record.get("evaluation_output_layer_timestamp") or time.time()),
                None
                if record.get("evaluation_output_layer_frame_id") is None
                else str(record.get("evaluation_output_layer_frame_id")),
                record.get("evaluation_output_layer_fps"),
                int(record.get("evaluation_output_layer_detection_count") or 0),
                int(record.get("evaluation_output_layer_track_count") or 0),
                int(record.get("evaluation_output_layer_vlm_call_count") or 0),
                int(record.get("evaluation_output_layer_scene_call_count") or 0),
                module_latency_json,
                record_json,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _safe_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 1


def _pluck(package: Any, key: str, default: Any = None) -> Any:
    if package is None:
        return default
    if isinstance(package, dict):
        return package.get(key, default)
    return getattr(package, key, default)


def _count_detections(yolo_layer_package: Any) -> int:
    detections = _pluck(yolo_layer_package, "yolo_layer_detections", None)
    return _safe_len(detections)


def _count_tracks(tracking_layer_package: Any) -> int:
    # Contract describes per-track fields; in practice this may be a list of records.
    if tracking_layer_package is None:
        return 0
    if isinstance(tracking_layer_package, list):
        return len(tracking_layer_package)
    tracks = _pluck(tracking_layer_package, "tracking_layer_tracks", None)
    if tracks is not None:
        return _safe_len(tracks)
    # Fallback: single record dict/object
    return 1


def _count_vlm_calls(vlm_layer_package: Any) -> int:
    if vlm_layer_package is None:
        return 0
    if isinstance(vlm_layer_package, list):
        return len(vlm_layer_package)
    calls = _pluck(vlm_layer_package, "vlm_layer_calls", None)
    if calls is not None:
        return _safe_len(calls)
    return 1


def _count_scene_calls(scene_awareness_layer_package: Any) -> int:
    if scene_awareness_layer_package is None:
        return 0
    if isinstance(scene_awareness_layer_package, list):
        return len(scene_awareness_layer_package)
    calls = _pluck(scene_awareness_layer_package, "scene_awareness_layer_calls", None)
    if calls is not None:
        return _safe_len(calls)
    return 1


def _estimate_fps(timings: dict | None) -> float | None:
    if not timings:
        return None

    for key in ("pipeline_total_s", "end_to_end_s", "frame_duration_s", "total_s"):
        v = timings.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return 1.0 / float(v)

    durations = timings.get("frame_durations_s")
    if isinstance(durations, (list, tuple)) and durations:
        vals = [float(x) for x in durations if isinstance(x, (int, float)) and x > 0]
        if vals:
            return 1.0 / (sum(vals) / len(vals))

    start = timings.get("frame_start_s")
    end = timings.get("frame_end_s")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
        return 1.0 / float(end - start)

    return None


def _extract_module_latency(timings: dict | None) -> dict:
    if not timings:
        return {}

    module_latency = timings.get("module_latency")
    if isinstance(module_latency, dict):
        return dict(module_latency)

    # Heuristic: accept any *_latency_s or *_latency_ms keys.
    out: dict[str, float] = {}
    for k, v in timings.items():
        if not isinstance(v, (int, float)):
            continue
        if k.endswith("_latency_s"):
            out[k.removesuffix("_latency_s")] = float(v)
        elif k.endswith("_latency_ms"):
            out[k.removesuffix("_latency_ms")] = float(v) / 1000.0
    return out


def collect_evaluation_metrics(
    input_layer_package=None,
    roi_layer_package=None,
    yolo_layer_package=None,
    tracking_layer_package=None,
    vlm_layer_package=None,
    scene_awareness_layer_package=None,
    timings: dict | None = None,
) -> dict:
    """
    Gather basic benchmarking / telemetry metrics from active layers.
    """
    metrics: dict[str, Any] = {}

    metrics["fps"] = _estimate_fps(timings)
    metrics["module_latency"] = _extract_module_latency(timings)

    metrics["detection_count"] = _count_detections(yolo_layer_package)
    metrics["track_count"] = _count_tracks(tracking_layer_package)
    metrics["vlm_call_count"] = _count_vlm_calls(vlm_layer_package)
    metrics["scene_call_count"] = _count_scene_calls(scene_awareness_layer_package)

    # Optional identifiers for debugging/joins (not part of required package fields).
    metrics["frame_id"] = (
        _pluck(input_layer_package, "input_layer_frame_id", None)
        or _pluck(roi_layer_package, "roi_layer_frame_id", None)
        or _pluck(yolo_layer_package, "yolo_layer_frame_id", None)
        or _pluck(tracking_layer_package, "tracking_layer_frame_id", None)
        or _pluck(scene_awareness_layer_package, "scene_awareness_layer_frame_id", None)
    )
    metrics["timestamp"] = (
        _pluck(input_layer_package, "input_layer_timestamp", None)
        or _pluck(roi_layer_package, "roi_layer_timestamp", None)
        or _pluck(scene_awareness_layer_package, "scene_awareness_layer_timestamp", None)
        or time.time()
    )

    return metrics


def build_evaluation_output_layer_package(metrics: dict) -> dict:
    """
    Create the `evaluation_output_layer_package` per the pipeline contract.
    """
    return {
        "evaluation_output_layer_fps": metrics.get("fps"),
        "evaluation_output_layer_module_latency": metrics.get("module_latency", {}) or {},
        "evaluation_output_layer_detection_count": int(metrics.get("detection_count", 0) or 0),
        "evaluation_output_layer_track_count": int(metrics.get("track_count", 0) or 0),
        "evaluation_output_layer_vlm_call_count": int(metrics.get("vlm_call_count", 0) or 0),
        "evaluation_output_layer_scene_call_count": int(metrics.get("scene_call_count", 0) or 0),
        # Extra debug fields (allowed; contract lists required fields, not exclusivity).
        "evaluation_output_layer_frame_id": metrics.get("frame_id"),
        "evaluation_output_layer_timestamp": metrics.get("timestamp"),
    }


def emit_evaluation_output(
    evaluation_output_layer_package,
    output_destination: str = "sqlite",
    output_path: str = "evaluation.sqlite",
) -> None:
    """
    Write evaluation records to the configured sink.

    Minimum viable sink: sqlite3 database file at `output_path`.
    """
    record = evaluation_output_layer_package
    if not isinstance(record, dict):
        raise TypeError("evaluation_output_layer_package must be a dict")

    if output_destination not in {"sqlite", "stdout"}:
        raise ValueError(f"Unsupported output_destination: {output_destination!r}")

    if output_destination == "stdout":
        print(json.dumps(record, separators=(",", ":"), ensure_ascii=False, sort_keys=True))
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    module_latency_json = json.dumps(
        record.get("evaluation_output_layer_module_latency", {}) or {},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    record_json = json.dumps(record, separators=(",", ":"), ensure_ascii=False)

    conn = sqlite3.connect(output_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_records (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_at_s REAL NOT NULL,
              frame_id TEXT,
              fps REAL,
              detection_count INTEGER NOT NULL,
              track_count INTEGER NOT NULL,
              vlm_call_count INTEGER NOT NULL,
              scene_call_count INTEGER NOT NULL,
              module_latency_json TEXT NOT NULL,
              evaluation_package_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO evaluation_records (
              created_at_s,
              frame_id,
              fps,
              detection_count,
              track_count,
              vlm_call_count,
              scene_call_count,
              module_latency_json,
              evaluation_package_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                float(record.get("evaluation_output_layer_timestamp") or time.time()),
                None if record.get("evaluation_output_layer_frame_id") is None else str(record.get("evaluation_output_layer_frame_id")),
                record.get("evaluation_output_layer_fps"),
                int(record.get("evaluation_output_layer_detection_count") or 0),
                int(record.get("evaluation_output_layer_track_count") or 0),
                int(record.get("evaluation_output_layer_vlm_call_count") or 0),
                int(record.get("evaluation_output_layer_scene_call_count") or 0),
                module_latency_json,
                record_json,
            ),
        )
        conn.commit()
    finally:
        conn.close()

