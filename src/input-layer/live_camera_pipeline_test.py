#!/usr/bin/env python3
"""
live_camera_pipeline_test.py

Lives in: src/input-layer/
Purpose: End-to-end integration test for the live camera path.

This script is owned by input_layer because the camera hardware is
input_layer's concern. The script demonstrates that a live camera
source flows correctly through the perception pipeline:

    camera -> input_layer -> yolo_layer -> tracking_layer -> display

Run from project root on the Jetson:
    python src/input-layer/live_camera_pipeline_test.py
    python src/input-layer/live_camera_pipeline_test.py --device 0 --conf 0.4

Options:
    --device N           Camera device index (default 0)
    --width, --height    Output resolution (default 1280x720)
    --conf X             YOLO confidence threshold (default 0.4)
    --device-compute     "cpu" or "cuda" for YOLO inference
    --use-gstreamer      Use GStreamer (for Jetson CSI cameras)

Controls (while window is focused):
    q  quit
    s  save screenshot to data/

NOTE: This test exercises three layers together. It is not a unit test
for any single layer. It lives with input_layer because the live camera
path is input_layer's responsibility.
"""

import argparse
import os
import sys
import time

# --- PATH SETUP ---
# Allow imports from sibling layer folders despite their hyphens.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_this_dir, "..", "..")
_yolo_dir = os.path.join(_project_root, "src", "yolo-layer")
_tracking_dir = os.path.join(_project_root, "src", "tracking-layer")
_input_dir = _this_dir

sys.path.insert(0, _input_dir)
sys.path.insert(0, _yolo_dir)
sys.path.insert(0, _tracking_dir)

import cv2

from input_layer import InputLayer
from detector import (
    initialize_yolo_layer,
    run_yolo_detection,
    filter_yolo_detections,
    build_yolo_layer_package,
)
from tracker import (
    initialize_tracking_layer,
    update_tracks,
    assign_tracking_status,
    build_tracking_layer_package,
)


# Colors in BGR
COLORS = {
    "new":    (0, 255, 0),    # green
    "active": (255, 180, 0),  # blue-ish
    "lost":   (0, 0, 255),    # red
}


def draw_tracks(frame, tracking_pkg):
    """
    Draw bounding boxes, IDs, and labels on the frame.

    Expects tracking_pkg in the list-of-dicts format:
        {
            "tracking_layer_frame_id": int,
            "tracking_layer_tracks": [
                {
                    "tracking_layer_track_id": int,
                    "tracking_layer_bbox": [x1, y1, x2, y2],
                    "tracking_layer_detector_class": str,
                    "tracking_layer_confidence": float,
                    "tracking_layer_status": "new" | "active" | "lost",
                },
                ...
            ],
        }
    """
    tracks = tracking_pkg.get("tracking_layer_tracks", [])

    for trk in tracks:
        status = trk.get("tracking_layer_status", "active")
        if status == "lost":
            continue  # skip lost tracks to reduce visual clutter

        color = COLORS.get(status, (200, 200, 200))

        # Bounding box
        bbox = trk["tracking_layer_bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        tid = trk["tracking_layer_track_id"]
        cls = trk.get("tracking_layer_detector_class", "?")
        conf = trk.get("tracking_layer_confidence", 0.0)
        label = f"ID={tid} {cls} {conf:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 2)

        label_y = y1 - 6 if y1 > 25 else y2 + th + 6
        cv2.rectangle(frame, (x1, label_y - th - 4),
                      (x1 + tw + 4, label_y + 4), color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y), font, font_scale,
                    (0, 0, 0), 2, cv2.LINE_AA)

    return frame


def draw_hud(frame, frame_id, tracking_pkg, fps_display):
    """Draw a stats overlay in the top-left corner."""
    tracks = tracking_pkg.get("tracking_layer_tracks", [])
    visible = sum(1 for t in tracks
                  if t.get("tracking_layer_status") != "lost")

    lines = [
        f"Frame: {frame_id}  |  FPS: {fps_display:.1f}",
        f"Visible tracks: {visible}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 25
    for line in lines:
        cv2.putText(frame, line, (12, y), font, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), font, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += 28

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Live camera pipeline test (input + yolo + tracking)"
    )
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Frame height (default: 720)")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLO model name or path")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="Confidence threshold")
    parser.add_argument("--device-compute", default="cpu",
                        help="Compute device for YOLO: cpu, cuda, cuda:0")
    parser.add_argument("--use-gstreamer", action="store_true",
                        help="Use GStreamer pipeline (Jetson CSI cameras)")
    args = parser.parse_args()

    # --- Initialize input layer ---
    print("Initializing input_layer (camera)...")
    input_layer = InputLayer()
    input_layer.initialize_input_layer(
        config_input_source="camera",
        config_frame_resolution=(args.width, args.height),
        camera_device_index=args.device,
        use_gstreamer=args.use_gstreamer,
    )
    print(f"[input_layer] device={args.device}  "
          f"resolution={args.width}x{args.height}  "
          f"gstreamer={args.use_gstreamer}")
    print()

    # --- Initialize yolo_layer ---
    initialize_yolo_layer(
        model_name=args.model,
        conf_threshold=args.conf,
        device=args.device_compute,
    )
    print()

    # --- Initialize tracking_layer ---
    initialize_tracking_layer(frame_rate=30)
    print()

    print("=" * 55)
    print("LIVE PIPELINE TEST — press 'q' to quit, 's' to screenshot")
    print("=" * 55)
    print()

    # --- Main loop ---
    frame_id = 0
    fps_display = 0.0
    prev_time = time.time()

    try:
        while True:
            # Layer 2: read frame via input_layer
            raw_frame = input_layer.read_next_frame()
            if raw_frame is None:
                print("WARNING: read_next_frame returned None, retrying...")
                continue

            input_pkg = input_layer.build_input_layer_package(raw_frame)

            # Convert dataclass to dict for downstream layers
            input_dict = {
                "input_layer_frame_id": input_pkg.input_layer_frame_id,
                "input_layer_timestamp": input_pkg.input_layer_timestamp,
                "input_layer_image": input_pkg.input_layer_image,
                "input_layer_source_type": input_pkg.input_layer_source_type,
                "input_layer_resolution": input_pkg.input_layer_resolution,
            }

            # Layer 4: YOLO
            raw_dets = run_yolo_detection(input_dict)
            filtered_dets = filter_yolo_detections(raw_dets)
            yolo_pkg = build_yolo_layer_package(
                input_dict["input_layer_frame_id"], filtered_dets
            )

            # Layer 5: Tracking
            current_tracks = update_tracks(yolo_pkg)
            status_tracks = assign_tracking_status(current_tracks, frame_id)
            tracking_pkg = build_tracking_layer_package(frame_id, status_tracks)

            # Draw and display
            display_frame = input_pkg.input_layer_image.copy()
            display_frame = draw_tracks(display_frame, tracking_pkg)
            display_frame = draw_hud(display_frame, frame_id,
                                     tracking_pkg, fps_display)

            cv2.imshow("Live Pipeline Test", display_frame)

            # FPS calc
            now = time.time()
            dt = now - prev_time
            fps_display = 1.0 / dt if dt > 0 else 0.0
            prev_time = now

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            elif key == ord("s"):
                shot_path = f"data/screenshot_frame_{frame_id}.jpg"
                cv2.imwrite(shot_path, display_frame)
                print(f"Screenshot saved: {shot_path}")

            frame_id += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        input_layer.close_input_layer()
        cv2.destroyAllWindows()
        print(f"Closed. Processed {frame_id} frames.")


if __name__ == "__main__":
    main()
