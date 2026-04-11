#!/usr/bin/env python3
"""
save_roi_videos.py

Batch-save ROI visualization videos for sample1.mp4 through sample4.mp4 using
the same rendering logic as visualize_roi.py.

Default output directory:
    E:\OneDrive\desktop\roi-optimization

Run from project root:
    python src/roi-layer/save_roi_videos.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from visualize_roi import (
    InputLayer,
    apply_roi_to_frame,
    build_roi_layer_package,
    build_yolo_layer_package,
    draw_detection_boxes,
    draw_roi_bounds,
    draw_text_block,
    filter_yolo_detections,
    initialize_roi_layer,
    initialize_yolo_layer,
    input_package_to_dict,
    load_runtime_settings,
    make_canvas,
    probe_video_metadata,
    run_yolo_detection,
    update_roi_state,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path(r"E:\OneDrive\desktop\roi-optimization")


def build_sample_video_paths() -> list[Path]:
    return [REPO_ROOT / "data" / f"sample{index}.mp4" for index in range(1, 5)]


def render_one_video(
    video_path: Path,
    output_path: Path,
    *,
    settings: dict[str, object],
    max_frames: int,
) -> None:
    frame_resolution = tuple(settings["frame_resolution"])
    fps, width, height = probe_video_metadata("video", str(video_path), frame_resolution)
    panel_width = max(360, width // 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width + panel_width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    input_layer = InputLayer()
    input_layer.initialize_input_layer(
        config_input_source="video",
        config_frame_resolution=frame_resolution,
        config_input_path=str(video_path),
    )
    initialize_yolo_layer(
        model_name=str(settings["model"]),
        conf_threshold=float(settings["conf"]),
        device=str(settings["device"]),
    )
    initialize_roi_layer(
        config_roi_enabled=bool(settings["roi_enabled"]),
        config_roi_vehicle_count_threshold=int(settings["roi_threshold"]),
    )

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
                    f"Threshold: {settings['roi_threshold']}",
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
                int(settings["roi_threshold"]),
                len(yolo_pkg["yolo_layer_detections"]),
            )
            writer.write(canvas)

            if max_frames > 0 and input_package.input_layer_frame_id >= max_frames:
                break
    finally:
        input_layer.close_input_layer()
        writer.release()


def main() -> None:
    defaults = load_runtime_settings()
    parser = argparse.ArgumentParser(description="Batch-save ROI visualization videos for sample1-4.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for saved ROI videos.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap per sample.")
    args = parser.parse_args()

    settings = {
        **defaults,
        "roi_enabled": True,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for video_path in build_sample_video_paths():
        if not video_path.is_file():
            print(f"[save-roi-videos] skipping missing file: {video_path}")
            continue
        output_path = output_dir / f"{video_path.stem}_roi_visualized.mp4"
        render_one_video(
            video_path,
            output_path,
            settings=settings,
            max_frames=int(args.max_frames),
        )
        print(f"[save-roi-videos] saved: {output_path}")


if __name__ == "__main__":
    main()
