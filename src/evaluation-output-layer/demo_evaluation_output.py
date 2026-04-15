from __future__ import annotations

import json
import os
from pathlib import Path

from evaluation_output_layer import (
    build_evaluation_output_layer_package,
    collect_evaluation_metrics,
    emit_evaluation_output,
)


def main() -> None:
    # Fake upstream packages (dict shape matches the contract doc).
    input_layer_package = {
        "input_layer_frame_id": "frame_000123",
        "input_layer_timestamp": 1713200000.123,
        "input_layer_source_type": "video",
        "input_layer_resolution": (1280, 720),
    }
    roi_layer_package = {
        "roi_layer_frame_id": "frame_000123",
        "roi_layer_timestamp": 1713200000.123,
        "roi_layer_enabled": True,
        "roi_layer_locked": True,
        "roi_layer_bounds": (100, 50, 1000, 650),
    }
    yolo_layer_package = {
        "yolo_layer_frame_id": "frame_000123",
        "yolo_layer_detections": [
            {"yolo_detection_bbox": (10, 20, 200, 140), "yolo_detection_class": "truck", "yolo_detection_confidence": 0.93},
            {"yolo_detection_bbox": (300, 40, 520, 220), "yolo_detection_class": "car", "yolo_detection_confidence": 0.88},
        ],
    }
    tracking_layer_package = [
        {
            "tracking_layer_frame_id": "frame_000123",
            "tracking_layer_track_id": 7,
            "tracking_layer_bbox": (12, 22, 202, 142),
            "tracking_layer_detector_class": "truck",
            "tracking_layer_confidence": 0.91,
            "tracking_layer_status": "active",
        },
        {
            "tracking_layer_frame_id": "frame_000123",
            "tracking_layer_track_id": 9,
            "tracking_layer_bbox": (301, 41, 519, 219),
            "tracking_layer_detector_class": "car",
            "tracking_layer_confidence": 0.87,
            "tracking_layer_status": "new",
        },
    ]
    vlm_layer_package = [
        {
            "vlm_layer_track_id": 7,
            "vlm_layer_query_type": "truck_type",
            "vlm_layer_label": "flatbed",
            "vlm_layer_attributes": {"has_trailer": True},
            "vlm_layer_confidence": 0.72,
            "vlm_layer_model_id": "demo-model",
        }
    ]
    scene_awareness_layer_package = {
        "scene_awareness_layer_frame_id": "frame_000123",
        "scene_awareness_layer_timestamp": 1713200000.123,
        "scene_awareness_layer_label": "loading_dock",
        "scene_awareness_layer_attributes": {"time_of_day": "day"},
        "scene_awareness_layer_confidence": 0.61,
    }

    timings = {
        "module_latency": {
            "input_layer": 0.004,
            "roi_layer": 0.002,
            "yolo_layer": 0.030,
            "tracking_layer": 0.006,
            "vlm_layer": 0.120,
            "scene_awareness_layer": 0.090,
        },
        "pipeline_total_s": 0.004 + 0.002 + 0.030 + 0.006 + 0.120 + 0.090,
    }

    metrics = collect_evaluation_metrics(
        input_layer_package=input_layer_package,
        roi_layer_package=roi_layer_package,
        yolo_layer_package=yolo_layer_package,
        tracking_layer_package=tracking_layer_package,
        vlm_layer_package=vlm_layer_package,
        scene_awareness_layer_package=scene_awareness_layer_package,
        timings=timings,
    )
    evaluation_pkg = build_evaluation_output_layer_package(metrics)

    # Print a single JSON line (easy to grep/log/pipe).
    print(json.dumps(evaluation_pkg, separators=(",", ":"), ensure_ascii=False))

    out_db = Path(__file__).parent / "evaluation_demo.sqlite"
    if out_db.exists():
        os.remove(out_db)
    emit_evaluation_output(evaluation_pkg, output_destination="sqlite", output_path=str(out_db))
    print(f"Wrote sqlite DB to: {out_db}")


if __name__ == "__main__":
    main()
