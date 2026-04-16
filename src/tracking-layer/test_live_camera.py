#!/usr/bin/env python3
"""
test_live_camera.py
Live camera test — runs YOLO + Tracking on a USB camera feed.

Uses the teammate's input_layer for camera access (no rule violation).

Run from project root on the Jetson:
    python src/tracking-layer/test_live_camera.py
    python src/tracking-layer/test_live_camera.py --device 0 --conf 0.4
    python src/tracking-layer/test_live_camera.py --use-gstreamer  (for CSI cameras)

Press 'q' to quit the live window.
Press 's' to save a screenshot.

NOTE: This script imports from input-layer and yolo-layer.
      This is a test/integration script living in tracking-layer.
      If this violates your team's rules, move it to pipeline/ instead.
"""

import argparse
import sys
import os
import time

# --- PATH SETUP ---
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_this_dir, "..", "..")
_yolo_dir = os.path.join(_project_root, "src", "yolo-layer")
_input_dir = os.path.join(_project_root, "src", "input-layer")
_tracking_dir = _this_dir

sys.path.insert(0, _yolo_dir)
sys.path.insert(0, _input_dir)
sys.path.insert(0, _tracking_dir)

import cv2
import numpy as np

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
    """Draw bounding boxes, IDs, and labels on the frame."""
    for trk in tracking_pkg["tracking_layer_tracks"]:
        status = trk["tracking_layer_status"]
        if status == "lost":
            continue  # skip lost tracks to reduce clutter on live feed

        color = COLORS.get(status, (200, 200, 200))

        # Bounding box
        x1, y1, x2, y2 = [int(v) for v in trk["tracking_layer_bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        tid = trk["tracking_layer_track_id"]
        cls = trk["tracking_layer_detector_class"]
        conf = trk["tracking_layer_confidence"]
        label = f"ID={tid} {cls} {conf:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 2)

        label_y = y1 - 6 if y1 > 25 else y2 + th + 6
        cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 4), color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y), font, font_scale, (0, 0, 0), 2, cv2.LINE_AA)

    return frame


def draw_hud(frame, frame_id, tracking_pkg, fps_display):
    """Draw stats overlay in the top-left corner."""
    tracks = tracking_pkg["tracking_layer_tracks"]
    visible = [t for t in tracks if t["tracking_layer_status"] != "lost"]

    lines = [
        f"Frame: {frame_id}  |  FPS: {fps_display:.1f}",
        f"Visible tracks: {len(visible)}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 25
    for line in lines:
        cv2.putText(frame, line, (12, y), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28

    return frame


def main():
    parser = argparse.ArgumentParser(description="Live camera tracking test")
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
                        help="Compute device: cpu, cuda, cuda:0")
    parser.add_argument("--use-gstreamer", action="store_true",
                        help="Use GStreamer pipeline (for Jetson CSI cameras)")
    args = parser.parse_args()

    # --- Initialize input layer (teammate's code) ---
    print("Initializing input layer (camera)...")
    input_layer = InputLayer()
    input_layer.initialize_input_layer(
        config_input_source="camera",
        config_frame_resolution=(args.width, args.height),
        camera_device_index=args.device,
        use_gstreamer=args.use_gstreamer,
    )
    print(f"Camera opened: device={args.device}, "
          f"resolution={args.width}x{args.height}, "
          f"gstreamer={args.use_gstreamer}")
    print()

    # --- Initialize YOLO layer ---
    initialize_yolo_layer(
        model_name=args.model,
        conf_threshold=args.conf,
        device=args.device_compute,
    )
    print()

    # --- Initialize tracking layer ---
    initialize_tracking_layer(frame_rate=30)
    print()

    print("=" * 50)
    print("LIVE TRACKING — press 'q' to quit, 's' to screenshot")
    print("=" * 50)
    print()

    # --- Main loop ---
    frame_id = 0
    fps_display = 0.0
    prev_time = time.time()

    try:
        while True:
            # Layer 2: Read frame via input layer
            raw_frame = input_layer.read_next_frame()
            if raw_frame is None:
                print("WARNING: Failed to read frame, retrying...")
                continue

            # Build input layer package
            input_pkg = input_layer.build_input_layer_package(raw_frame)

            # Convert InputLayerPackage to dict for our layers
            # (teammate uses a dataclass, our layers expect a dict)
            input_dict = {
                "input_layer_frame_id": input_pkg.input_layer_frame_id,
                "input_layer_timestamp": input_pkg.input_layer_timestamp,
                "input_layer_image": input_pkg.input_layer_image,
                "input_layer_source_type": input_pkg.input_layer_source_type,
                "input_layer_resolution": input_pkg.input_layer_resolution,
            }

            # Layer 4: YOLO detection
            raw_dets = run_yolo_detection(input_dict)
            filtered_dets = filter_yolo_detections(raw_dets)
            yolo_pkg = build_yolo_layer_package(
                input_dict["input_layer_frame_id"], filtered_dets
            )

            # Layer 5: Tracking
            current_tracks = update_tracks(yolo_pkg)
            status_tracks = assign_tracking_status(current_tracks, frame_id)
            tracking_pkg = build_tracking_layer_package(frame_id, status_tracks)

            # Draw results on frame
            display_frame = input_pkg.input_layer_image.copy()
            display_frame = draw_tracks(display_frame, tracking_pkg)
            display_frame = draw_hud(display_frame, frame_id, tracking_pkg, fps_display)

            # Show live window
            cv2.imshow("Live Tracking", display_frame)

            # FPS calculation
            now = time.time()
            fps_display = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
            prev_time = now

            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            elif key == ord("s"):
                screenshot_path = f"data/screenshot_frame_{frame_id}.jpg"
                cv2.imwrite(screenshot_path, display_frame)
                print(f"Screenshot saved: {screenshot_path}")

            frame_id += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        input_layer.close_input_layer()
        cv2.destroyAllWindows()
        print(f"Closed. Processed {frame_id} frames.")


if __name__ == "__main__":
    main()