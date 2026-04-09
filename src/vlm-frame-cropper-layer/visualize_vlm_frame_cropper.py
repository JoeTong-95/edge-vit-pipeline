#!/usr/bin/env python3
"""
visualize_vlm_frame_cropper.py
Visualize tracked vehicle crops on a canvas alongside frame overlays.

Run from src/vlm-frame-cropper-layer:
    python .\visualize_vlm_frame_cropper.py
    python .\visualize_vlm_frame_cropper.py --show

Run from project root:
    python src/vlm-frame-cropper-layer/visualize_vlm_frame_cropper.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _THIS_DIR.parent / "configuration-layer"
_INPUT_DIR = _THIS_DIR.parent / "input-layer"
_YOLO_DIR = _THIS_DIR.parent / "yolo-layer"
_TRACKING_DIR = _THIS_DIR.parent / "tracking-layer"
_REPO_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_CONFIG_DIR))
sys.path.insert(0, str(_INPUT_DIR))
sys.path.insert(0, str(_YOLO_DIR))
sys.path.insert(0, str(_TRACKING_DIR))

import cv2
import numpy as np
from config_node import get_config_value, load_config, validate_config
from detector import build_yolo_layer_package, filter_yolo_detections, initialize_yolo_layer, run_yolo_detection
from input_layer import InputLayer
from tracker import build_tracking_layer_package, initialize_tracking_layer, assign_tracking_status, update_tracks
from vlm_frame_cropper_layer import build_vlm_frame_cropper_package, build_vlm_frame_cropper_request_package, extract_vlm_object_crop

STATUS_COLORS = {
    "new": (0, 255, 0),
    "active": (255, 180, 0),
    "lost": (0, 0, 255),
}
FRAME_TEXT = (255, 255, 255)
SHADOW_TEXT = (0, 0, 0)



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
        "vlm_enabled": get_config_value(config, "config_vlm_enabled"),
        "output": str((_REPO_ROOT / "data" / "vlm_frame_cropper_visualization.mp4").resolve()),
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



def tracking_rows(tracking_pkg):
    return [
        {
            "track_id": track_id,
            "bbox": bbox,
            "detector_class": detector_class,
            "confidence": confidence,
            "status": status,
        }
        for track_id, bbox, detector_class, confidence, status in zip(
            tracking_pkg["tracking_layer_track_id"],
            tracking_pkg["tracking_layer_bbox"],
            tracking_pkg["tracking_layer_detector_class"],
            tracking_pkg["tracking_layer_confidence"],
            tracking_pkg["tracking_layer_status"],
        )
    ]



def draw_text_block(frame, lines, origin=(10, 30), scale=0.6):
    x, y = origin
    for line in lines:
        cv2.putText(frame, line, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, SHADOW_TEXT, 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, FRAME_TEXT, 2, cv2.LINE_AA)
        y += int(28 * scale / 0.6)



def draw_tracking_overlay(frame, tracking_pkg):
    for row in tracking_rows(tracking_pkg):
        color = STATUS_COLORS.get(row["status"], (200, 200, 200))
        x1, y1, x2, y2 = [int(value) for value in row["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {row['track_id']} {row['detector_class']} {row['status']}"
        cv2.putText(frame, label, (x1 + 2, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)



def build_crop_contact_sheet(crops, frame_height, panel_width):
    panel = np.zeros((frame_height, panel_width, 3), dtype=np.uint8)
    draw_text_block(panel, ["TRACK CROPS"], origin=(10, 28), scale=0.7)
    if not crops:
        draw_text_block(panel, ["No active/new tracked vehicles passed through"], origin=(10, 70), scale=0.55)
        return panel

    tile_w = (panel_width - 30) // 2
    tile_h = max(100, (frame_height - 110) // 3)
    x_positions = [10, 20 + tile_w]
    y = 60
    column = 0

    for crop_info in crops:
        crop = crop_info["crop"]
        resized = cv2.resize(crop, (tile_w, tile_h))
        x = x_positions[column]
        panel[y:y + tile_h, x:x + tile_w] = resized
        color = STATUS_COLORS.get(crop_info["status"], (255, 255, 255))
        cv2.rectangle(panel, (x, y), (x + tile_w, y + tile_h), color, 2)
        draw_text_block(
            panel,
            [
                f"ID {crop_info['track_id']} {crop_info['status']}",
                f"bbox {tuple(int(v) for v in crop_info['bbox'])}",
                f"crop {crop.shape[1]}x{crop.shape[0]}",
            ],
            origin=(x + 4, y + 20),
            scale=0.45,
        )
        column += 1
        if column == 2:
            column = 0
            y += tile_h + 20
        if y + tile_h > frame_height:
            break

    return panel



def main():
    defaults = load_runtime_settings()
    parser = argparse.ArgumentParser(description="Visualize VLM frame cropper outputs")
    parser.add_argument("--input-source", default=defaults["input_source"], choices=["video", "camera"])
    parser.add_argument("--video", default=defaults["video"])
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--conf", type=float, default=defaults["conf"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--gstreamer", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", default=defaults["output"])
    args = parser.parse_args()

    frame_resolution = tuple(defaults["frame_resolution"])
    fps, width, height = probe_video_metadata(args.input_source, args.video, frame_resolution)
    panel_width = max(420, width // 2)
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width + panel_width, height))
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
    initialize_tracking_layer(frame_rate=int(fps))

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
            current_tracks = update_tracks(yolo_pkg)
            status_tracks = assign_tracking_status(current_tracks, input_package.input_layer_frame_id)
            tracking_pkg = build_tracking_layer_package(input_package.input_layer_frame_id, status_tracks)

            frame_view = input_package.input_layer_image.copy()
            draw_tracking_overlay(frame_view, tracking_pkg)
            elapsed = time.time() - start_time
            fps_text = input_package.input_layer_frame_id / elapsed if elapsed > 0 else 0.0
            draw_text_block(
                frame_view,
                [
                    f"Frame: {input_package.input_layer_frame_id}",
                    f"Tracked objects: {len(tracking_pkg['tracking_layer_track_id'])}",
                    f"VLM enabled flag: {defaults['vlm_enabled']}",
                    f"FPS: {fps_text:.1f}",
                ],
            )

            crops = []
            for index, row in enumerate(tracking_rows(tracking_pkg)):
                if row["status"] == "lost":
                    continue
                request_pkg = build_vlm_frame_cropper_request_package(
                    input_layer_package=input_package,
                    tracking_layer_package=tracking_pkg,
                    track_index=index,
                    vlm_frame_cropper_trigger_reason=f"tracking_status:{row['status']}",
                    config_vlm_enabled=True,
                )
                crop = extract_vlm_object_crop(input_package, request_pkg)
                crop_pkg = build_vlm_frame_cropper_package(request_pkg, crop)
                crops.append(
                    {
                        "track_id": crop_pkg["vlm_frame_cropper_layer_track_id"],
                        "bbox": crop_pkg["vlm_frame_cropper_layer_bbox"],
                        "status": row["status"],
                        "crop": crop_pkg["vlm_frame_cropper_layer_image"],
                    }
                )

            panel = build_crop_contact_sheet(crops, height, panel_width)
            canvas = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
            canvas[:, :width] = frame_view
            canvas[:, width:] = panel
            out.write(canvas)

            if args.show:
                cv2.imshow("VLM Frame Cropper Visualizer", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames > 0 and input_package.input_layer_frame_id >= args.max_frames:
                break
    finally:
        input_layer.close_input_layer()
        out.release()
        cv2.destroyAllWindows()

    print(f"Saved VLM frame cropper visualization to: {args.output}")


if __name__ == "__main__":
    main()
