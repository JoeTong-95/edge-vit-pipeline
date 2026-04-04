#!/usr/bin/env python3
"""
test_yolo_layer.py
Test script for Layer 4 (YOLO Detection) in isolation.

Run from src/yolo-layer/:
    cd src/yolo-layer
    python test_yolo_layer.py --video ../../data/sample.mp4

Or from project root:
    python src/yolo-layer/test_yolo_layer.py --video data/sample.mp4

This verifies:
  - YOLO model loads correctly
  - detections are filtered to target classes (truck/car/bus)
  - yolo_layer_package format matches the spec
  - confidence threshold is reasonable for your video
"""

import argparse
import sys
import os
import json

# Make sure imports work when running from this folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from detector import (
    initialize_yolo_layer,
    run_yolo_detection,
    filter_yolo_detections,
    build_yolo_layer_package,
)


def build_fake_input_layer_package(frame, frame_id, fps):
    """
    Build an input_layer_package matching the Layer 2 (Input) spec.
    In the real pipeline, your teammate's input layer creates these.
    Here we create them manually for testing.
    """
    return {
        "input_layer_frame_id": frame_id,
        "input_layer_timestamp": round(frame_id / fps, 4),
        "input_layer_image": frame,
        "input_layer_source_type": "video",
        "input_layer_resolution": (frame.shape[1], frame.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser(description="Test YOLO layer on a video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model name or path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="cpu", help="Device: cpu, cuda, cuda:0")
    parser.add_argument("--max-frames", type=int, default=100,
                        help="Stop after N frames (0 = process all)")
    parser.add_argument("--save-json", default="",
                        help="Save all detection packets to this JSON file")
    args = parser.parse_args()

    # --- Open video ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.video}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    print()

    # --- Initialize YOLO layer ---
    initialize_yolo_layer(
        model_name=args.model,
        conf_threshold=args.conf,
        device=args.device,
    )
    print()

    # --- Process frames ---
    frame_id = 0
    total_detections = 0
    all_packages = []  # for optional JSON export

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Build the upstream package (normally comes from input_layer)
        input_pkg = build_fake_input_layer_package(frame, frame_id, fps)

        # Run the full YOLO pipeline: detect -> filter -> package
        raw_dets = run_yolo_detection(input_pkg)
        filtered_dets = filter_yolo_detections(raw_dets)
        yolo_pkg = build_yolo_layer_package(frame_id, filtered_dets)

        # Print frames that have detections
        n_dets = len(yolo_pkg["yolo_layer_detections"])
        if n_dets > 0:
            total_detections += n_dets
            print(f"Frame {frame_id}: {n_dets} detection(s)")
            for det in yolo_pkg["yolo_layer_detections"]:
                bbox = [round(v, 1) for v in det["yolo_detection_bbox"]]
                print(f"  {det['yolo_detection_class']} "
                      f"(conf={det['yolo_detection_confidence']:.2f}) "
                      f"bbox={bbox}")

        if args.save_json:
            all_packages.append(yolo_pkg)

        frame_id += 1
        if args.max_frames > 0 and frame_id >= args.max_frames:
            break

    cap.release()

    # --- Summary ---
    print()
    print(f"Processed {frame_id} frames.")
    print(f"Total detections (target classes only): {total_detections}")

    # --- Optional: save to JSON for inspection or tracking layer testing ---
    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(all_packages, f, indent=2)
        print(f"Detection packages saved to: {args.save_json}")


if __name__ == "__main__":
    main()
