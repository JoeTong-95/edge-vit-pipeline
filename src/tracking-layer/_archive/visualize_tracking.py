#!/usr/bin/env python3
"""
visualize_tracking.py
Draw bounding boxes, track IDs, and labels onto a video.

Run from project root inside Docker:
    python src/tracking-layer/visualize_tracking.py --video data/sample.mp4
    python src/tracking-layer/visualize_tracking.py --video data/sample.mp4 --output data/tracked_output.mp4

Produces an annotated MP4 video you can open and watch.
Color coding:
    GREEN  = new (first appearance)
    BLUE   = active (continuing track)
    RED    = lost (disappeared, showing last known position)
"""

import argparse
import sys
import os

_this_dir = os.path.dirname(os.path.abspath(__file__))
_yolo_dir = os.path.join(_this_dir, "..", "yolo-layer")
sys.path.insert(0, _this_dir)
sys.path.insert(0, _yolo_dir)

import cv2
import numpy as np
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


# Colors in BGR (OpenCV format)
COLORS = {
    "new":    (0, 255, 0),    # green
    "active": (255, 180, 0),  # blue-ish
    "lost":   (0, 0, 255),    # red
}


def draw_tracks(frame, tracking_pkg):
    """
    Draw bounding boxes, track IDs, class labels, and confidence
    onto the frame. Returns the annotated frame.
    """
    for trk in tracking_pkg["tracking_layer_tracks"]:
        status = trk["tracking_layer_status"]
        color = COLORS.get(status, (200, 200, 200))

        # Bounding box
        x1, y1, x2, y2 = [int(v) for v in trk["tracking_layer_bbox"]]
        thickness = 3 if status != "lost" else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label: "ID=7 truck 0.85"
        tid = trk["tracking_layer_track_id"]
        cls = trk["tracking_layer_detector_class"]
        conf = trk["tracking_layer_confidence"]
        label = f"ID={tid} {cls} {conf:.2f}"

        # Background rectangle for readability
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        # Put label above the box, or below if too close to top
        label_y = y1 - 8 if y1 > 30 else y2 + text_h + 8
        label_x = x1

        cv2.rectangle(
            frame,
            (label_x, label_y - text_h - 4),
            (label_x + text_w + 4, label_y + 4),
            color, -1,  # filled rectangle
        )
        cv2.putText(
            frame, label,
            (label_x + 2, label_y),
            font, font_scale, (0, 0, 0),  # black text
            font_thickness, cv2.LINE_AA,
        )

    return frame


def draw_hud(frame, frame_id, tracking_pkg, fps_actual):
    """Draw a heads-up display with frame count and track summary."""
    tracks = tracking_pkg["tracking_layer_tracks"]
    new_count = sum(1 for t in tracks if t["tracking_layer_status"] == "new")
    active_count = sum(1 for t in tracks if t["tracking_layer_status"] == "active")
    lost_count = sum(1 for t in tracks if t["tracking_layer_status"] == "lost")

    lines = [
        f"Frame: {frame_id}",
        f"FPS: {fps_actual:.1f}",
        f"Tracks: {new_count} new | {active_count} active | {lost_count} lost",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30
    for line in lines:
        # Shadow for readability
        cv2.putText(frame, line, (12, y_offset), font, 0.7,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y_offset), font, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 30

    return frame


def main():
    parser = argparse.ArgumentParser(description="Visualize tracking on video")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", default="data/tracked_output.mp4",
                        help="Output annotated video path")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--device", default="cpu", help="Device: cpu, cuda")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames (0 = all)")
    parser.add_argument("--skip-lost", action="store_true",
                        help="Don't draw lost tracks")
    args = parser.parse_args()

    # --- Open input video ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input:  {args.video}")
    print(f"Output: {args.output}")
    print(f"Size:   {width}x{height} @ {fps:.1f} FPS")
    print(f"Frames: {total_frames}")
    print()

    # --- Create output video writer ---
    # Use mp4v codec — works in Docker without extra installs
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"ERROR: Could not create output video: {args.output}")
        sys.exit(1)

    # --- Initialize layers ---
    initialize_yolo_layer(model_name=args.model, conf_threshold=args.conf,
                          device=args.device)
    print()
    initialize_tracking_layer(frame_rate=int(fps))
    print()

    # --- Process frames ---
    import time
    frame_id = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Build input package
        input_pkg = {
            "input_layer_frame_id": frame_id,
            "input_layer_timestamp": round(frame_id / fps, 4),
            "input_layer_image": frame,
            "input_layer_source_type": "video",
            "input_layer_resolution": (width, height),
        }

        # Layer 4: YOLO
        raw_dets = run_yolo_detection(input_pkg)
        filtered_dets = filter_yolo_detections(raw_dets)
        yolo_pkg = build_yolo_layer_package(frame_id, filtered_dets)

        # Layer 5: Tracking
        current_tracks = update_tracks(yolo_pkg)
        status_tracks = assign_tracking_status(current_tracks, frame_id)
        tracking_pkg = build_tracking_layer_package(frame_id, status_tracks)

        # Optionally filter out lost tracks from the visualization
        if args.skip_lost:
            tracking_pkg["tracking_layer_tracks"] = [
                t for t in tracking_pkg["tracking_layer_tracks"]
                if t["tracking_layer_status"] != "lost"
            ]

        # Draw on frame
        elapsed = time.time() - start_time
        fps_actual = (frame_id + 1) / elapsed if elapsed > 0 else 0

        annotated = frame.copy()
        annotated = draw_tracks(annotated, tracking_pkg)
        annotated = draw_hud(annotated, frame_id, tracking_pkg, fps_actual)

        # Write frame
        out.write(annotated)

        # Progress
        if frame_id % 50 == 0:
            n_tracks = len(tracking_pkg["tracking_layer_tracks"])
            print(f"Frame {frame_id}/{total_frames} — "
                  f"{n_tracks} tracks — {fps_actual:.1f} FPS")

        frame_id += 1
        if args.max_frames > 0 and frame_id >= args.max_frames:
            break

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    print()
    print(f"Done. Processed {frame_id} frames in {elapsed:.1f}s")
    print(f"Average: {frame_id / elapsed:.1f} FPS")
    print(f"Output saved to: {args.output}")
    print()
    print(f"Open the file to watch: {args.output}")


if __name__ == "__main__":
    main()
