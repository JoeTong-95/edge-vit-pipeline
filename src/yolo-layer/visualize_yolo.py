#!/usr/bin/env python3
"""
visualize_yolo.py
Draw YOLO detections onto a normalized input-layer stream without tracking.

Run from src/yolo-layer:
    python .\visualize_yolo.py
    python .\visualize_yolo.py --show
    python .\visualize_yolo.py --save-metrics

Run from project root:
    python src/yolo-layer/visualize_yolo.py
    python src/yolo-layer/visualize_yolo.py --show

CLI arguments override values loaded from src/configuration-layer/config.yaml.
Produces an annotated MP4 video and can optionally show a live preview window.
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
_CONFIG_DIR = _THIS_DIR.parent / "configuration-layer"
_INPUT_DIR = _THIS_DIR.parent / "input-layer"
_REPO_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_CONFIG_DIR))
sys.path.insert(0, str(_INPUT_DIR))

import cv2
from config_node import get_config_value, load_config, validate_config
from detector import build_yolo_layer_package, filter_yolo_detections, initialize_yolo_layer, run_yolo_detection
from input_layer import InputLayer

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)
HUD_TEXT_COLOR = (255, 255, 255)
HUD_SHADOW_COLOR = (0, 0, 0)


def update_metric_ema(metrics, metric_name, sample_value, alpha=0.2):
    if metric_name not in metrics:
        metrics[metric_name] = sample_value
        return
    metrics[metric_name] = (alpha * sample_value) + ((1.0 - alpha) * metrics[metric_name])



def infer_model_family(model_name):
    model_text = Path(str(model_name)).name.lower()
    if "yolov11" in model_text or "yolo11" in model_text or "v11" in model_text:
        return "v11"
    if "yolov10" in model_text or "yolo10" in model_text or "v10" in model_text:
        return "v10"
    if "yolov8" in model_text or "yolo8" in model_text or "v8" in model_text:
        return "v8"
    return "unknown"



def initialize_metrics_db(db_path):
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS visualizer_runs (
            run_id TEXT PRIMARY KEY,
            visualizer_type TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            video_path TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_family TEXT NOT NULL,
            device_mode TEXT NOT NULL,
            confidence REAL NOT NULL,
            show_window INTEGER NOT NULL,
            output_path TEXT NOT NULL,
            source_fps REAL NOT NULL,
            frame_width INTEGER NOT NULL,
            frame_height INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS visualizer_frames (
            frame_record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            frame_id INTEGER NOT NULL,
            elapsed_seconds REAL NOT NULL,
            fps_actual REAL NOT NULL,
            average_fps REAL NOT NULL,
            infer_ms REAL NOT NULL,
            infer_fps REAL NOT NULL,
            average_infer_fps REAL NOT NULL,
            count_total INTEGER NOT NULL,
            count_new INTEGER,
            count_active INTEGER,
            count_lost INTEGER,
            FOREIGN KEY(run_id) REFERENCES visualizer_runs(run_id)
        )
        """
    )
    connection.commit()
    return connection



def insert_run_record(connection, args, fps, width, height):
    run_id = str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO visualizer_runs (
            run_id, visualizer_type, created_at_utc, video_path, model_name,
            model_family, device_mode, confidence, show_window, output_path,
            source_fps, frame_width, frame_height
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            "yolo",
            datetime.now(timezone.utc).isoformat(),
            args.video,
            args.model,
            infer_model_family(args.model),
            args.device,
            float(args.conf),
            int(args.show),
            args.output,
            float(fps),
            int(width),
            int(height),
        ),
    )
    connection.commit()
    return run_id



def insert_frame_record(connection, run_id, frame_id, elapsed_seconds, fps_actual, average_fps, infer_ms, infer_fps, average_infer_fps, total_count):
    connection.execute(
        """
        INSERT INTO visualizer_frames (
            run_id, frame_id, elapsed_seconds, fps_actual, average_fps,
            infer_ms, infer_fps, average_infer_fps, count_total,
            count_new, count_active, count_lost
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
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
            int(total_count),
        ),
    )



def draw_detections(frame, yolo_pkg):
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



def draw_hud(frame, frame_id, yolo_pkg, fps_actual, average_fps, infer_fps, average_infer_fps, device_mode):
    lines = [
        f"Frame: {frame_id}",
        f"Mode: {device_mode}",
        f"FPS: {fps_actual:.1f} | Avg: {average_fps:.1f}",
        f"Infer FPS: {infer_fps:.1f} | Avg: {average_infer_fps:.1f}",
        f"Detections: {len(yolo_pkg['yolo_layer_detections'])}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30
    for line in lines:
        cv2.putText(frame, line, (12, y_offset), font, 0.7, HUD_SHADOW_COLOR, 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y_offset), font, 0.7, HUD_TEXT_COLOR, 2, cv2.LINE_AA)
        y_offset += 30
    return frame



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
    return {
        "input_source": get_config_value(config, "config_input_source"),
        "video": _resolve_repo_path(get_config_value(config, "config_input_path")),
        "frame_resolution": tuple(get_config_value(config, "config_frame_resolution")),
        "model": get_config_value(config, "config_yolo_model"),
        "conf": get_config_value(config, "config_yolo_confidence_threshold"),
        "device": get_config_value(config, "config_device"),
        "output": "",
        "metrics_db": str((_REPO_ROOT / "data" / "visualizer_metrics.sqlite").resolve()),
    }



def main():
    runtime_defaults = load_runtime_settings()
    parser = argparse.ArgumentParser(description="Visualize YOLO detections on input-layer output")
    parser.add_argument("--input-source", default=runtime_defaults["input_source"], choices=["video", "camera"], help="Input source selected through the input layer")
    parser.add_argument("--video", default=runtime_defaults["video"], help="Input video path used when --input-source is video")
    parser.add_argument("--model", default=runtime_defaults["model"], help="YOLO model")
    parser.add_argument("--conf", type=float, default=runtime_defaults["conf"], help="Confidence threshold")
    parser.add_argument("--device", default=runtime_defaults["device"], help="Device: cpu, cuda")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index used when --input-source is camera")
    parser.add_argument("--gstreamer", action="store_true", help="Use GStreamer camera pipeline when --input-source is camera")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all)")
    parser.add_argument("--show", action="store_true", help="Show a live preview window while processing")
    parser.add_argument("--save-metrics", action="store_true", help="Save run metadata and frame metrics to a SQLite database")
    parser.add_argument("--metrics-db", default=runtime_defaults["metrics_db"], help="SQLite database path used when --save-metrics is enabled")
    parser.add_argument("--output", default=runtime_defaults["output"], help="Optional output annotated video path")
    args = parser.parse_args()

    frame_resolution = tuple(runtime_defaults["frame_resolution"])
    fps, width, height, total_frames = probe_source_metadata(args.input_source, args.video, frame_resolution)

    print(f"Input source: {args.input_source}")
    if args.input_source == "video":
        print(f"Input path:   {args.video}")
    print(f"Output:       {args.output}")
    print(f"Model:        {args.model}")
    print(f"Family:       {infer_model_family(args.model)}")
    print(f"Device:       {args.device}")
    print(f"Conf:         {args.conf}")
    print(f"Size:         {width}x{height} @ {fps:.1f} FPS")
    print(f"Frames:       {total_frames if total_frames > 0 else 'unknown'}")
    if args.save_metrics:
        print(f"Metrics DB:   {args.metrics_db}")
    print()

    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_db) if os.path.dirname(args.metrics_db) else ".", exist_ok=True)
    out = None
    if args.output:
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not out.isOpened():
            print(f"ERROR: Could not create output video: {args.output}")
            sys.exit(1)

    metrics_connection = None
    run_id = None
    if args.save_metrics:
        metrics_connection = initialize_metrics_db(args.metrics_db)
        run_id = insert_run_record(metrics_connection, args, fps, width, height)

    input_layer = InputLayer()
    input_layer.initialize_input_layer(
        config_input_source=args.input_source,
        config_frame_resolution=frame_resolution,
        config_input_path=args.video,
        camera_device_index=args.camera_index,
        use_gstreamer=args.gstreamer,
    )

    initialize_yolo_layer(model_name=args.model, conf_threshold=args.conf, device=args.device)
    print()

    processed_frames = 0
    start_time = time.time()
    metrics = {}
    total_infer_seconds = 0.0

    try:
        while True:
            loop_start = time.perf_counter()
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
            update_metric_ema(metrics, "infer_ms", infer_ms)
            total_infer_seconds += infer_ms / 1000.0

            elapsed = time.time() - start_time
            fps_actual = processed_frames / elapsed if elapsed > 0 else 0.0
            average_fps = fps_actual
            infer_fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
            average_infer_fps = processed_frames / total_infer_seconds if total_infer_seconds > 0 else 0.0

            draw_start = time.perf_counter()
            annotated = input_package.input_layer_image.copy()
            annotated = draw_detections(annotated, yolo_pkg)
            annotated = draw_hud(annotated, frame_id, yolo_pkg, fps_actual, average_fps, infer_fps, average_infer_fps, args.device)
            update_metric_ema(metrics, "draw_ms", (time.perf_counter() - draw_start) * 1000.0)

            write_start = time.perf_counter()
            if out is not None:
                out.write(annotated)
            update_metric_ema(metrics, "write_ms", (time.perf_counter() - write_start) * 1000.0)

            if args.show:
                cv2.imshow("YOLO Visualizer", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Preview stopped by user.")
                    break

            update_metric_ema(metrics, "loop_ms", (time.perf_counter() - loop_start) * 1000.0)

            if metrics_connection is not None and run_id is not None:
                insert_frame_record(metrics_connection, run_id, frame_id, elapsed, fps_actual, average_fps, infer_ms, infer_fps, average_infer_fps, len(yolo_pkg["yolo_layer_detections"]))

            if frame_id % 50 == 0:
                total_frames_text = total_frames if total_frames > 0 else "unknown"
                print(f"Frame {frame_id}/{total_frames_text} - {len(yolo_pkg['yolo_layer_detections'])} detections - {fps_actual:.1f} FPS - inferFPS {average_infer_fps:.1f} - infer {metrics.get('infer_ms', 0.0):.1f} ms - loop {metrics.get('loop_ms', 0.0):.1f} ms")

            if args.max_frames > 0 and frame_id >= args.max_frames:
                break
            processed_frames = frame_id
    finally:
        input_layer.close_input_layer()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        if metrics_connection is not None:
            metrics_connection.commit()
            metrics_connection.close()

    elapsed = time.time() - start_time
    print()
    print(f"Done. Processed {processed_frames} frames in {elapsed:.1f}s")
    if elapsed > 0 and processed_frames > 0:
        print(f"Average: {processed_frames / elapsed:.1f} FPS")
    print(f"Output saved to: {args.output}")
    if args.save_metrics:
        print(f"Metrics saved to: {args.metrics_db}")


if __name__ == "__main__":
    main()
