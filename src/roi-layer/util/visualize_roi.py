#!/usr/bin/env python3
"""
visualize_roi.py
Visualize ROI discovery, ROI lock state, and the active cropped region.

Run from project root:
    python src/roi-layer/util/visualize_roi.py
    python src/roi-layer/util/visualize_roi.py --show
"""

import argparse
import os
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROI_DIR = _THIS_DIR.parent
_SRC_DIR = _ROI_DIR.parent
_CONFIG_DIR = _SRC_DIR / "configuration-layer"
_INPUT_DIR = _SRC_DIR / "input-layer"
_YOLO_DIR = _SRC_DIR / "yolo-layer"
_REPO_ROOT = _SRC_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_ROI_DIR))
sys.path.insert(0, str(_CONFIG_DIR))
sys.path.insert(0, str(_INPUT_DIR))
sys.path.insert(0, str(_YOLO_DIR))

import cv2
import numpy as np
from config_node import get_config_value, load_config, validate_config
from detector import build_yolo_layer_package, filter_yolo_detections, initialize_yolo_layer, run_yolo_detection
from input_layer import InputLayer
from roi_layer import apply_roi_to_frame, build_roi_layer_package, initialize_roi_layer, update_roi_state

FRAME_TEXT = (255, 255, 255)
SHADOW_TEXT = (0, 0, 0)
ROI_COLOR = (0, 200, 255)
DETECTION_COLOR = (0, 255, 0)
LOCKED_COLOR = (0, 220, 120)
PANEL_BG = (24, 24, 24)
TEXT_SUPERSAMPLE = 2



def _resolve_repo_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())



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
        "roi_enabled": get_config_value(config, "config_roi_enabled"),
        "roi_threshold": get_config_value(config, "config_roi_vehicle_count_threshold"),
        "output": "",
    }



def probe_video_metadata(input_source, input_path, frame_resolution):
    width, height = frame_resolution
    fps = 30.0
    if input_source == "video":
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not probe video source: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    return fps, width, height



def input_package_to_dict(package):
    return {
        "input_layer_frame_id": package.input_layer_frame_id,
        "input_layer_timestamp": package.input_layer_timestamp,
        "input_layer_image": package.input_layer_image,
        "input_layer_source_type": package.input_layer_source_type,
        "input_layer_resolution": package.input_layer_resolution,
    }



def draw_text_block(frame, lines, origin=(10, 30), color=FRAME_TEXT, scale=0.48, line_gap=24, shadow_offset=2):
    if not lines:
        return

    h, w = frame.shape[:2]
    super_h = h * TEXT_SUPERSAMPLE
    super_w = w * TEXT_SUPERSAMPLE
    overlay = np.zeros((super_h, super_w, 3), dtype=np.uint8)

    x, y = origin
    x *= TEXT_SUPERSAMPLE
    y *= TEXT_SUPERSAMPLE
    super_scale = scale * TEXT_SUPERSAMPLE
    super_gap = line_gap * TEXT_SUPERSAMPLE
    super_shadow_offset = shadow_offset * TEXT_SUPERSAMPLE

    for line in lines:
        cv2.putText(
            overlay,
            line,
            (x + super_shadow_offset, y + super_shadow_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            super_scale,
            SHADOW_TEXT,
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            super_scale,
            color,
            2,
            cv2.LINE_AA,
        )
        y += super_gap

    overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
    mask = np.any(overlay > 0, axis=2)
    frame[mask] = overlay[mask]



def draw_detection_boxes(frame, detections):
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection["yolo_detection_bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), DETECTION_COLOR, 2)



def draw_roi_bounds(frame, roi_bounds, locked):
    if roi_bounds is None:
        return
    x1, y1, x2, y2 = [int(value) for value in roi_bounds]
    thickness = 3 if locked else 2
    color = LOCKED_COLOR if locked else ROI_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = "ROI LOCKED" if locked else "ROI CANDIDATE"
    draw_text_block(frame, [label], origin=(x1 + 4, max(20, y1 - 10)), color=color, scale=0.34, line_gap=18, shadow_offset=1)



def make_canvas(full_frame, roi_frame, roi_pkg, roi_state, target_count, current_detection_count):
    full_h, full_w = full_frame.shape[:2]
    panel_w = max(360, full_w // 2)
    canvas = np.zeros((full_h, full_w + panel_w, 3), dtype=np.uint8)
    canvas[:, :full_w] = full_frame
    canvas[:, full_w:] = PANEL_BG

    candidate_count = roi_state["roi_candidate_box_count"]
    progress_count = min(candidate_count, target_count) if target_count > 0 else candidate_count
    progress_text = f"Detection: {progress_count}/{target_count}" if target_count > 0 else f"Detection: {progress_count}"
    status_text = "Locked" if roi_pkg["roi_layer_locked"] else "Searching"
    status_color = LOCKED_COLOR if roi_pkg["roi_layer_locked"] else ROI_COLOR

    info_origin_x = full_w + 14
    info_origin_y = 28
    draw_text_block(canvas, ["ROI VIEW"], origin=(info_origin_x, info_origin_y), scale=0.62, line_gap=28)
    draw_text_block(
        canvas,
        [
            f"Enabled: {roi_pkg['roi_layer_enabled']}",
            f"Locked: {roi_pkg['roi_layer_locked']}",
            progress_text,
            f"Status: {status_text}",
            f"Frame detections: {current_detection_count}",
            f"Bounds: {roi_pkg['roi_layer_bounds']}",
        ],
        origin=(info_origin_x, info_origin_y + 42),
        scale=0.50,
        line_gap=26,
    )
    draw_text_block(
        canvas,
        [f"Status: {status_text}"],
        origin=(info_origin_x, info_origin_y + 42 + (3 * 26)),
        color=status_color,
        scale=0.50,
        line_gap=26,
    )

    preview_top = 220
    preview_bottom_margin = 18
    preview_h_available = max(80, full_h - preview_top - preview_bottom_margin)
    preview_w_available = panel_w - 28

    if roi_frame.size > 0:
        roi_h, roi_w = roi_frame.shape[:2]
        scale = min(preview_w_available / roi_w, preview_h_available / roi_h)
        preview_w = max(1, int(roi_w * scale))
        preview_h = max(1, int(roi_h * scale))
        preview = cv2.resize(roi_frame, (preview_w, preview_h))
        x_offset = full_w + 14 + (preview_w_available - preview_w) // 2
        y_offset = preview_top + (preview_h_available - preview_h) // 2
        canvas[y_offset:y_offset + preview.shape[0], x_offset:x_offset + preview.shape[1]] = preview
        cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + preview.shape[1], y_offset + preview.shape[0]), status_color, 2)

    return canvas



def main():
    defaults = load_runtime_settings()
    parser = argparse.ArgumentParser(description="Visualize ROI discovery and cropping")
    parser.add_argument("--input-source", default=defaults["input_source"], choices=["video", "camera"])
    parser.add_argument("--video", default=defaults["video"])
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--conf", type=float, default=defaults["conf"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--roi-enabled", action="store_true", default=defaults["roi_enabled"])
    parser.add_argument("--roi-threshold", type=int, default=defaults["roi_threshold"])
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--gstreamer", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", default=defaults["output"], help="Optional output video path")
    args = parser.parse_args()

    frame_resolution = tuple(defaults["frame_resolution"])
    fps, width, height = probe_video_metadata(args.input_source, args.video, frame_resolution)
    out = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width + max(360, width // 2), height))
        if not out.isOpened():
            raise RuntimeError(f"Could not create output video: {args.output}")

    input_layer = InputLayer()
    input_layer.initialize_input_layer(
        config_input_source=args.input_source,
        config_frame_resolution=frame_resolution,
        config_input_path=args.video,
        camera_device_index=args.camera_index,
        use_gstreamer=args.gstreamer,
    )
    initialize_yolo_layer(model_name=args.model, conf_threshold=args.conf, device=args.device)
    initialize_roi_layer(config_roi_enabled=args.roi_enabled, config_roi_vehicle_count_threshold=args.roi_threshold)

    start_time = time.time()
    try:
        while True:
            raw_frame = input_layer.read_next_frame()
            if raw_frame is None:
                break

            input_package = input_layer.build_input_layer_package(raw_frame)
            input_pkg = input_package_to_dict(input_package)
            raw_dets = run_yolo_detection(input_pkg)
            filtered_dets = filter_yolo_detections(raw_dets)
            yolo_pkg = build_yolo_layer_package(input_package.input_layer_frame_id, filtered_dets)
            roi_state = update_roi_state(input_package, yolo_pkg["yolo_layer_detections"])
            roi_frame = apply_roi_to_frame(input_package)
            roi_pkg = build_roi_layer_package(input_package, roi_frame)

            annotated = input_package.input_layer_image.copy()
            draw_detection_boxes(annotated, yolo_pkg["yolo_layer_detections"])
            draw_roi_bounds(annotated, roi_state["roi_layer_bounds"], roi_state["roi_layer_locked"])
            elapsed = time.time() - start_time
            fps_text = input_package.input_layer_frame_id / elapsed if elapsed > 0 else 0.0
            draw_text_block(
                annotated,
                [
                    f"Frame: {input_package.input_layer_frame_id}",
                    f"ROI enabled: {roi_pkg['roi_layer_enabled']}",
                    f"ROI locked: {roi_pkg['roi_layer_locked']}",
                    f"Threshold: {args.roi_threshold}",
                    f"FPS: {fps_text:.1f}",
                ],
                origin=(12, 26),
                scale=0.46,
                line_gap=24,
            )

            canvas = make_canvas(
                annotated,
                roi_frame,
                roi_pkg,
                roi_state,
                args.roi_threshold,
                len(yolo_pkg["yolo_layer_detections"]),
            )
            if out is not None:
                out.write(canvas)

            if args.show:
                cv2.imshow("ROI Visualizer", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames > 0 and input_package.input_layer_frame_id >= args.max_frames:
                break
    finally:
        input_layer.close_input_layer()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

    if args.output:
        print(f"Saved ROI visualization to: {args.output}")


if __name__ == "__main__":
    main()
