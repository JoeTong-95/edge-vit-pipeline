#!/usr/bin/env python3
"""
automated_evaluation.py
Run sequential evaluation sweeps across device, model family, and tracking mode.

Default combinations:
    - device: cpu, cuda
    - model: v8, v10, v11
    - tracking: off, on

Outputs:
    - annotated MP4 videos, one per run
    - SQLite database with one run row per combination and frame-level metrics

Examples:
    python .\automated_evaluation.py
    python .\automated_evaluation.py --max-seconds 60
    python .\automated_evaluation.py --max-frames 300
    python .\automated_evaluation.py --output-dir "E:\OneDrive\desktop\video"
"""

import argparse
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_TRACKING_DIR = _THIS_DIR.parent
_SRC_DIR = _TRACKING_DIR.parent
_YOLO_DIR = _SRC_DIR / "yolo-layer"
_CONFIG_DIR = _SRC_DIR / "configuration-layer"
_INPUT_DIR = _SRC_DIR / "input-layer"
_REPO_ROOT = _SRC_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_TRACKING_DIR))
sys.path.insert(0, str(_YOLO_DIR))
sys.path.insert(0, str(_CONFIG_DIR))
sys.path.insert(0, str(_INPUT_DIR))

import cv2
from config_node import get_config_value, load_config, validate_config
from detector import build_yolo_layer_package, filter_yolo_detections, initialize_yolo_layer, run_yolo_detection
from input_layer import InputLayer
from tracker import assign_tracking_status, build_tracking_layer_package, initialize_tracking_layer, update_tracks

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)
HUD_TEXT_COLOR = (255, 255, 255)
HUD_SHADOW_COLOR = (0, 0, 0)
TRACK_COLORS = {
    "new": (0, 255, 0),
    "active": (255, 180, 0),
    "lost": (0, 0, 255),
}
MODEL_MAP = {
    "v8": "yolov8n.pt",
    "v10": "yolov10n.pt",
    "v11": "yolo11n.pt",
}


def infer_model_family(model_name):
    model_text = Path(str(model_name)).name.lower()
    if "yolov11" in model_text or "yolo11" in model_text or "v11" in model_text:
        return "v11"
    if "yolov10" in model_text or "yolo10" in model_text or "v10" in model_text:
        return "v10"
    if "yolov8" in model_text or "yolo8" in model_text or "v8" in model_text:
        return "v8"
    return "unknown"



def tracking_package_rows(tracking_pkg):
    return [
        {
            "tracking_layer_track_id": track_id,
            "tracking_layer_bbox": bbox,
            "tracking_layer_detector_class": detector_class,
            "tracking_layer_confidence": confidence,
            "tracking_layer_status": status,
        }
        for track_id, bbox, detector_class, confidence, status in zip(
            tracking_pkg["tracking_layer_track_id"],
            tracking_pkg["tracking_layer_bbox"],
            tracking_pkg["tracking_layer_detector_class"],
            tracking_pkg["tracking_layer_confidence"],
            tracking_pkg["tracking_layer_status"],
        )
    ]



def draw_yolo_detections(frame, yolo_pkg):
    for det in yolo_pkg["yolo_layer_detections"]:
        x1, y1, x2, y2 = [int(v) for v in det["yolo_detection_bbox"]]
        label = f"{det['yolo_detection_class']} {det['yolo_detection_confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        label_y = y1 - 8 if y1 > 30 else y2 + text_h + 8
        label_x = x1
        cv2.rectangle(frame, (label_x, label_y - text_h - 4), (label_x + text_w + 4, label_y + 4), BOX_COLOR, -1)
        cv2.putText(frame, label, (label_x + 2, label_y), font, font_scale, TEXT_COLOR, font_thickness, cv2.LINE_AA)
    return frame



def draw_tracking_detections(frame, tracking_pkg):
    for track in tracking_package_rows(tracking_pkg):
        status = track["tracking_layer_status"]
        color = TRACK_COLORS.get(status, (200, 200, 200))
        x1, y1, x2, y2 = [int(v) for v in track["tracking_layer_bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if status != "lost" else 1)
        label = f"ID={track['tracking_layer_track_id']} {track['tracking_layer_detector_class']} {track['tracking_layer_confidence']:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        label_y = y1 - 8 if y1 > 30 else y2 + text_h + 8
        label_x = x1
        cv2.rectangle(frame, (label_x, label_y - text_h - 4), (label_x + text_w + 4, label_y + 4), color, -1)
        cv2.putText(frame, label, (label_x + 2, label_y), font, font_scale, TEXT_COLOR, font_thickness, cv2.LINE_AA)
    return frame



def draw_run_hud(frame, run_label, frame_id, device_mode, fps_actual, average_fps, infer_fps, average_infer_fps, detections_count, tracks_count):
    lines = [
        f"Run: {run_label}",
        f"Frame: {frame_id}",
        f"Mode: {device_mode}",
        f"FPS: {fps_actual:.1f} | Avg: {average_fps:.1f}",
        f"Infer FPS: {infer_fps:.1f} | Avg: {average_infer_fps:.1f}",
        f"Detections: {detections_count} | Tracks: {tracks_count}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30
    for line in lines:
        cv2.putText(frame, line, (12, y_offset), font, 0.7, HUD_SHADOW_COLOR, 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y_offset), font, 0.7, HUD_TEXT_COLOR, 2, cv2.LINE_AA)
        y_offset += 30
    return frame



def _ensure_table_column(connection, table_name, column_name, column_definition):
    existing_columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})")}
    if column_name not in existing_columns:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")



def initialize_metrics_db(db_path):
    connection = sqlite3.connect(db_path, timeout=30.0)
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            video_path TEXT NOT NULL,
            output_video_path TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_family TEXT NOT NULL,
            device_mode TEXT NOT NULL,
            tracking_enabled INTEGER NOT NULL,
            confidence REAL NOT NULL,
            source_fps REAL NOT NULL,
            frame_width INTEGER NOT NULL,
            frame_height INTEGER NOT NULL,
            max_frames_requested INTEGER NOT NULL,
            max_seconds_requested REAL NOT NULL,
            status TEXT NOT NULL,
            processed_frames INTEGER,
            elapsed_seconds REAL,
            average_fps REAL,
            average_infer_fps REAL,
            average_detections REAL,
            average_tracks REAL
        )
        """
    )
    _ensure_table_column(connection, "evaluation_runs", "max_seconds_requested", "REAL NOT NULL DEFAULT 0")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_frames (
            frame_record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            frame_id INTEGER NOT NULL,
            elapsed_seconds REAL NOT NULL,
            fps_actual REAL NOT NULL,
            average_fps REAL NOT NULL,
            infer_ms REAL NOT NULL,
            infer_fps REAL NOT NULL,
            average_infer_fps REAL NOT NULL,
            detections_count INTEGER NOT NULL,
            tracks_count INTEGER NOT NULL,
            new_count INTEGER NOT NULL,
            active_count INTEGER NOT NULL,
            lost_count INTEGER NOT NULL,
            FOREIGN KEY(run_id) REFERENCES evaluation_runs(run_id)
        )
        """
    )
    connection.commit()
    return connection



def insert_run_record(connection, combo, fps, width, height, max_frames_requested, max_seconds_requested):
    run_id = str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO evaluation_runs (
            run_id, created_at_utc, video_path, output_video_path, model_name,
            model_family, device_mode, tracking_enabled, confidence, source_fps,
            frame_width, frame_height, max_frames_requested, max_seconds_requested, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            datetime.now(timezone.utc).isoformat(),
            combo["video_path"],
            combo["output_video_path"],
            combo["model_name"],
            combo["model_family"],
            combo["device_mode"],
            int(combo["tracking_enabled"]),
            float(combo["confidence"]),
            float(fps),
            int(width),
            int(height),
            int(max_frames_requested),
            float(max_seconds_requested),
            "running",
        ),
    )
    connection.commit()
    return run_id



def insert_frame_record(connection, run_id, frame_id, elapsed_seconds, fps_actual, average_fps, infer_ms, infer_fps, average_infer_fps, detections_count, tracks_count, new_count, active_count, lost_count):
    connection.execute(
        """
        INSERT INTO evaluation_frames (
            run_id, frame_id, elapsed_seconds, fps_actual, average_fps,
            infer_ms, infer_fps, average_infer_fps, detections_count,
            tracks_count, new_count, active_count, lost_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            int(frame_id),
            float(elapsed_seconds),
            float(fps_actual),
            float(average_fps),
            float(infer_ms),
            float(infer_fps),
            float(average_infer_fps),
            int(detections_count),
            int(tracks_count),
            int(new_count),
            int(active_count),
            int(lost_count),
        ),
    )



def finalize_run_record(connection, run_id, status, processed_frames, elapsed_seconds, average_fps, average_infer_fps, average_detections, average_tracks):
    connection.execute(
        """
        UPDATE evaluation_runs
        SET status = ?,
            processed_frames = ?,
            elapsed_seconds = ?,
            average_fps = ?,
            average_infer_fps = ?,
            average_detections = ?,
            average_tracks = ?
        WHERE run_id = ?
        """,
        (
            status,
            int(processed_frames),
            float(elapsed_seconds),
            float(average_fps),
            float(average_infer_fps),
            float(average_detections),
            float(average_tracks),
            run_id,
        ),
    )
    connection.commit()



def _resolve_repo_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())



def probe_source_metadata(input_source, input_path, frame_resolution):
    width, height = frame_resolution
    fps = 30.0
    total_frames = 0
    if input_source == "video":
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not probe video source: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return fps, width, height, total_frames



def input_package_to_dict(package):
    return {
        "input_layer_frame_id": package.input_layer_frame_id,
        "input_layer_timestamp": package.input_layer_timestamp,
        "input_layer_image": package.input_layer_image,
        "input_layer_source_type": package.input_layer_source_type,
        "input_layer_resolution": package.input_layer_resolution,
    }



def load_runtime_settings():
    config_path = _CONFIG_DIR / "config.yaml"
    config = load_config(config_path)
    validate_config(config)
    default_output_dir = Path(r"E:\OneDrive\desktop\video")
    return {
        "input_source": get_config_value(config, "config_input_source"),
        "video": _resolve_repo_path(get_config_value(config, "config_input_path")),
        "frame_resolution": tuple(get_config_value(config, "config_frame_resolution")),
        "confidence": get_config_value(config, "config_yolo_confidence_threshold"),
        "device": get_config_value(config, "config_device"),
        "output_dir": str(default_output_dir),
        "metrics_db": str((default_output_dir / "tracking_eval_metrics.sqlite").resolve()),
    }



def build_run_combinations(input_source, input_path, frame_resolution, output_dir, confidence, camera_index, use_gstreamer):
    combinations = []
    for device_mode in ["cpu", "cuda"]:
        for model_family, model_name in MODEL_MAP.items():
            for tracking_enabled in [False, True]:
                tracking_label = "tracking" if tracking_enabled else "notracking"
                output_name = f"eval_{model_family}_{device_mode}_{tracking_label}.mp4"
                combinations.append(
                    {
                        "input_source": input_source,
                        "video_path": input_path,
                        "frame_resolution": tuple(frame_resolution),
                        "camera_index": camera_index,
                        "use_gstreamer": use_gstreamer,
                        "output_video_path": str((Path(output_dir) / output_name).resolve()),
                        "model_name": model_name,
                        "model_family": model_family,
                        "device_mode": device_mode,
                        "tracking_enabled": tracking_enabled,
                        "confidence": confidence,
                        "run_label": f"{model_family.upper()} | {device_mode.upper()} | {tracking_label}",
                    }
                )
    return combinations



def run_single_evaluation(combo, connection, max_frames_requested, max_seconds_requested):
    fps, width, height, total_frames = probe_source_metadata(combo["input_source"], combo["video_path"], combo["frame_resolution"])

    os.makedirs(os.path.dirname(combo["output_video_path"]) or ".", exist_ok=True)
    out = cv2.VideoWriter(combo["output_video_path"], cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not create output video: {combo['output_video_path']}")

    run_id = insert_run_record(connection, combo, fps, width, height, max_frames_requested, max_seconds_requested)
    print()
    print(f"[eval] Starting {combo['run_label']}")
    print(f"[eval] Source: {combo['input_source']}")
    if combo["input_source"] == "video":
        print(f"[eval] Video:  {combo['video_path']}")
    print(f"[eval] Output: {combo['output_video_path']}")
    print(f"[eval] Frames: {total_frames if total_frames > 0 else 'unknown'}")
    print(f"[eval] Resolution: {width}x{height}")

    input_layer = InputLayer()
    input_layer.initialize_input_layer(
        config_input_source=combo["input_source"],
        config_frame_resolution=combo["frame_resolution"],
        config_input_path=combo["video_path"],
        camera_device_index=combo["camera_index"],
        use_gstreamer=combo["use_gstreamer"],
    )

    initialize_yolo_layer(model_name=combo["model_name"], conf_threshold=combo["confidence"], device=combo["device_mode"])
    if combo["tracking_enabled"]:
        initialize_tracking_layer(frame_rate=int(fps))

    processed_frames = 0
    start_time = time.time()
    total_infer_seconds = 0.0
    total_detections = 0
    total_tracks = 0

    try:
        while True:
            raw_frame = input_layer.read_next_frame()
            if raw_frame is None:
                break

            input_package = input_layer.build_input_layer_package(raw_frame)
            input_pkg = input_package_to_dict(input_package)
            frame_id = input_package.input_layer_frame_id

            infer_start = time.perf_counter()
            raw_dets = run_yolo_detection(input_pkg)
            filtered_dets = filter_yolo_detections(raw_dets)
            yolo_pkg = build_yolo_layer_package(frame_id, filtered_dets)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
            total_infer_seconds += infer_ms / 1000.0

            detections_count = len(yolo_pkg["yolo_layer_detections"])
            total_detections += detections_count

            tracks_count = 0
            new_count = 0
            active_count = 0
            lost_count = 0
            annotated = input_package.input_layer_image.copy()

            if combo["tracking_enabled"]:
                current_tracks = update_tracks(yolo_pkg)
                status_tracks = assign_tracking_status(current_tracks, frame_id)
                tracking_pkg = build_tracking_layer_package(frame_id, status_tracks)
                track_rows = tracking_package_rows(tracking_pkg)
                tracks_count = len(track_rows)
                new_count = sum(1 for track in track_rows if track["tracking_layer_status"] == "new")
                active_count = sum(1 for track in track_rows if track["tracking_layer_status"] == "active")
                lost_count = sum(1 for track in track_rows if track["tracking_layer_status"] == "lost")
                total_tracks += tracks_count
                annotated = draw_tracking_detections(annotated, tracking_pkg)
            else:
                annotated = draw_yolo_detections(annotated, yolo_pkg)

            elapsed = time.time() - start_time
            fps_actual = processed_frames / elapsed if elapsed > 0 else 0.0
            average_fps = fps_actual
            infer_fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
            average_infer_fps = processed_frames / total_infer_seconds if total_infer_seconds > 0 else 0.0

            annotated = draw_run_hud(annotated, combo["run_label"], frame_id, combo["device_mode"], fps_actual, average_fps, infer_fps, average_infer_fps, detections_count, tracks_count)
            out.write(annotated)

            insert_frame_record(connection, run_id, frame_id, elapsed, fps_actual, average_fps, infer_ms, infer_fps, average_infer_fps, detections_count, tracks_count, new_count, active_count, lost_count)

            if frame_id % 50 == 0:
                total_frames_text = total_frames if total_frames > 0 else "unknown"
                print(f"[eval] {combo['run_label']} frame {frame_id}/{total_frames_text} - FPS {fps_actual:.1f} - inferFPS {average_infer_fps:.1f} - det {detections_count} - tracks {tracks_count}")

            processed_frames = frame_id
            if max_frames_requested > 0 and frame_id >= max_frames_requested:
                break
            if max_seconds_requested > 0 and elapsed >= max_seconds_requested:
                break

        elapsed_total = time.time() - start_time
        average_fps = (processed_frames / elapsed_total) if elapsed_total > 0 and processed_frames > 0 else 0.0
        average_infer_fps = (processed_frames / total_infer_seconds) if total_infer_seconds > 0 and processed_frames > 0 else 0.0
        average_detections = (total_detections / processed_frames) if processed_frames > 0 else 0.0
        average_tracks = (total_tracks / processed_frames) if processed_frames > 0 else 0.0
        finalize_run_record(connection, run_id, "completed", processed_frames, elapsed_total, average_fps, average_infer_fps, average_detections, average_tracks)
        print(f"[eval] Completed {combo['run_label']} -> {combo['output_video_path']}")
    except Exception:
        elapsed_total = time.time() - start_time
        average_fps = (processed_frames / elapsed_total) if elapsed_total > 0 and processed_frames > 0 else 0.0
        average_infer_fps = (processed_frames / total_infer_seconds) if total_infer_seconds > 0 and processed_frames > 0 else 0.0
        average_detections = (total_detections / processed_frames) if processed_frames > 0 else 0.0
        average_tracks = (total_tracks / processed_frames) if processed_frames > 0 else 0.0
        finalize_run_record(connection, run_id, "failed", processed_frames, elapsed_total, average_fps, average_infer_fps, average_detections, average_tracks)
        raise
    finally:
        connection.commit()
        input_layer.close_input_layer()
        out.release()



def _fallback_metrics_db_path(metrics_db_path):
    metrics_path = Path(metrics_db_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(metrics_path.with_name(f"{metrics_path.stem}_{timestamp}{metrics_path.suffix}"))



def main():
    runtime_defaults = load_runtime_settings()
    parser = argparse.ArgumentParser(description="Run automated evaluation across model/device/tracking combinations")
    parser.add_argument("--input-source", default=runtime_defaults["input_source"], choices=["video", "camera"], help="Input source selected through the input layer")
    parser.add_argument("--video", default=runtime_defaults["video"], help="Input video path used when --input-source is video")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index used when --input-source is camera")
    parser.add_argument("--gstreamer", action="store_true", help="Use GStreamer camera pipeline when --input-source is camera")
    parser.add_argument("--conf", type=float, default=runtime_defaults["confidence"], help="Confidence threshold")
    parser.add_argument("--output-dir", default=runtime_defaults["output_dir"], help="Directory for annotated output videos")
    parser.add_argument("--metrics-db", default=runtime_defaults["metrics_db"], help="SQLite database path for evaluation metrics")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop each run after N frames (0 = all)")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Stop each run after N seconds of processed time (0 = all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics_db = Path(args.metrics_db)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metrics_db.parent if str(metrics_db.parent) else ".", exist_ok=True)

    combinations = build_run_combinations(args.input_source, args.video, runtime_defaults["frame_resolution"], output_dir, args.conf, args.camera_index, args.gstreamer)
    print(f"[eval] Scheduled {len(combinations)} runs")
    print(f"[eval] Output dir: {output_dir}")
    print(f"[eval] Metrics DB: {metrics_db}")
    if args.max_seconds > 0:
        print(f"[eval] Per-run duration limit: {args.max_seconds:.1f}s")

    resolved_metrics_db = str(metrics_db)
    try:
        connection = initialize_metrics_db(resolved_metrics_db)
    except sqlite3.OperationalError as exc:
        if "locked" not in str(exc).lower():
            raise
        fallback_metrics_db = _fallback_metrics_db_path(resolved_metrics_db)
        print(f"[eval] Metrics DB locked, using fallback DB: {fallback_metrics_db}")
        resolved_metrics_db = fallback_metrics_db
        connection = initialize_metrics_db(resolved_metrics_db)

    try:
        for index, combo in enumerate(combinations, start=1):
            print(f"[eval] Run {index}/{len(combinations)}")
            run_single_evaluation(combo, connection, args.max_frames, args.max_seconds)
    finally:
        connection.commit()
        connection.close()
        cv2.destroyAllWindows()

    print()
    print("[eval] All runs completed.")
    print(f"[eval] Videos saved to: {output_dir}")
    print(f"[eval] Metrics saved to: {resolved_metrics_db}")


if __name__ == "__main__":
    main()
