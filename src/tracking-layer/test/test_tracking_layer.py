#!/usr/bin/env python3
"""
test_tracking_layer.py
Test script for Layer 5 (Tracking) — tests tracking in isolation and
in combination with Layer 4 (YOLO).

Run from src/tracking-layer/:
    cd src/tracking-layer
    python test_tracking_layer.py --video ../../data/sample.mp4

Or from project root:
    python src/tracking-layer/test/test_tracking_layer.py --video data/sample.mp4

Modes:
    --video      Run detection + tracking on a live video (default mode)
    --from-json  Run tracking on saved yolo_layer_packages from a JSON file
                 (useful for testing tracking without re-running YOLO)

This verifies:
    - ByteTrack assigns persistent IDs across frames
    - new/active/lost status is correct
    - tracking_layer_package format matches the spec
"""

import argparse
import json
import sys
import os

# Make sure imports work after this test moved under tracking-layer/test.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_tracking_dir = os.path.abspath(os.path.join(_this_dir, ".."))
_src_dir = os.path.abspath(os.path.join(_tracking_dir, ".."))
_yolo_dir = os.path.join(_src_dir, "yolo-layer")
sys.path.insert(0, _tracking_dir)
sys.path.insert(0, _yolo_dir)

import cv2
from tracker import (
    initialize_tracking_layer,
    update_tracks,
    assign_tracking_status,
    build_tracking_layer_package,
)


def tracking_package_rows(tracking_pkg):
    """Convert the tracking_layer_package into per-track rows for display."""
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


def build_fake_input_layer_package(frame, frame_id, fps):
    """Build an input_layer_package for testing (normally Layer 2 does this)."""
    return {
        "input_layer_frame_id": frame_id,
        "input_layer_timestamp": round(frame_id / fps, 4),
        "input_layer_image": frame,
        "input_layer_source_type": "video",
        "input_layer_resolution": (frame.shape[1], frame.shape[0]),
    }


def run_video_mode(args):
    """Run YOLO detection + tracking on a video file."""
    from detector import (
        initialize_yolo_layer,
        run_yolo_detection,
        filter_yolo_detections,
        build_yolo_layer_package,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {args.video}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    print()

    initialize_yolo_layer(model_name=args.model, conf_threshold=args.conf,
                          device=args.device)
    print()
    initialize_tracking_layer(frame_rate=int(fps))
    print()

    frame_id = 0
    all_track_ids = set()
    all_packages = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_pkg = build_fake_input_layer_package(frame, frame_id, fps)

        raw_dets = run_yolo_detection(input_pkg)
        filtered_dets = filter_yolo_detections(raw_dets)
        yolo_pkg = build_yolo_layer_package(frame_id, filtered_dets)

        current_tracks = update_tracks(yolo_pkg)
        status_tracks = assign_tracking_status(current_tracks, frame_id)
        tracking_pkg = build_tracking_layer_package(frame_id, status_tracks)

        tracks = tracking_package_rows(tracking_pkg)
        if tracks:
            new_count = sum(1 for t in tracks if t["tracking_layer_status"] == "new")
            active_count = sum(1 for t in tracks if t["tracking_layer_status"] == "active")
            lost_count = sum(1 for t in tracks if t["tracking_layer_status"] == "lost")

            status_summary = []
            if new_count:
                status_summary.append(f"{new_count} new")
            if active_count:
                status_summary.append(f"{active_count} active")
            if lost_count:
                status_summary.append(f"{lost_count} lost")

            print(f"Frame {frame_id}: {' | '.join(status_summary)}")
            for trk in tracks:
                if trk["tracking_layer_status"] != "lost":
                    bbox = [round(v, 1) for v in trk["tracking_layer_bbox"]]
                    print(f"  ID={trk['tracking_layer_track_id']} "
                          f"{trk['tracking_layer_detector_class']} "
                          f"[{trk['tracking_layer_status']}] "
                          f"conf={trk['tracking_layer_confidence']:.2f} "
                          f"bbox={bbox}")

            for trk in tracks:
                all_track_ids.add(trk["tracking_layer_track_id"])

        if args.save_json:
            all_packages.append(tracking_pkg)

        frame_id += 1
        if args.max_frames > 0 and frame_id >= args.max_frames:
            break

    cap.release()

    print()
    print(f"Processed {frame_id} frames.")
    print(f"Unique track IDs seen: {sorted(all_track_ids)}")
    print(f"Total unique objects tracked: {len(all_track_ids)}")

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(all_packages, f, indent=2)
        print(f"Tracking packages saved to: {args.save_json}")


def run_json_mode(args):
    """Run tracking on pre-saved yolo_layer_packages from a JSON file."""
    print(f"Loading detections from: {args.from_json}")
    with open(args.from_json, "r") as f:
        yolo_packages = json.load(f)

    print(f"Loaded {len(yolo_packages)} detection packages.")
    print()

    initialize_tracking_layer()
    print()

    all_track_ids = set()

    for yolo_pkg in yolo_packages:
        frame_id = yolo_pkg["yolo_layer_frame_id"]

        current_tracks = update_tracks(yolo_pkg)
        status_tracks = assign_tracking_status(current_tracks, frame_id)
        tracking_pkg = build_tracking_layer_package(frame_id, status_tracks)

        tracks = tracking_package_rows(tracking_pkg)
        if tracks:
            visible = [t for t in tracks if t["tracking_layer_status"] != "lost"]
            if visible:
                print(f"Frame {frame_id}: {len(visible)} visible track(s)")
                for trk in visible:
                    bbox = [round(v, 1) for v in trk["tracking_layer_bbox"]]
                    print(f"  ID={trk['tracking_layer_track_id']} "
                          f"{trk['tracking_layer_detector_class']} "
                          f"[{trk['tracking_layer_status']}] "
                          f"conf={trk['tracking_layer_confidence']:.2f} "
                          f"bbox={bbox}")

            for trk in tracks:
                all_track_ids.add(trk["tracking_layer_track_id"])

    print()
    print(f"Processed {len(yolo_packages)} frames from JSON.")
    print(f"Unique track IDs: {sorted(all_track_ids)}")
    print(f"Total unique objects tracked: {len(all_track_ids)}")


def main():
    parser = argparse.ArgumentParser(description="Test tracking layer")
    parser.add_argument("--video", default="", help="Path to input video file")
    parser.add_argument("--from-json", default="",
                        help="Path to saved yolo_layer_packages JSON")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model (video mode)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="cpu", help="Device: cpu, cuda")
    parser.add_argument("--max-frames", type=int, default=100,
                        help="Stop after N frames, 0=all (video mode only)")
    parser.add_argument("--save-json", default="",
                        help="Save tracking packages to JSON file")
    args = parser.parse_args()

    if args.from_json:
        run_json_mode(args)
    elif args.video:
        run_video_mode(args)
    else:
        print("ERROR: Provide either --video or --from-json")
        print("Examples:")
        print("  python test_tracking_layer.py --video ../../data/sample.mp4")
        print("  python test_tracking_layer.py --from-json ../../data/detections.json")
        sys.exit(1)


if __name__ == "__main__":
    main()
