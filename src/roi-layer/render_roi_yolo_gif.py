#!/usr/bin/env python3
"""
render_roi_yolo_gif.py

Render the ROI calibration -> ROI lock -> ROI-cropped YOLO sequence into a GIF.

This script is visual-only. It does not write benchmark SQLite rows or charts.

Default behavior:
- uses the video path from config.yaml
- saves a GIF to E:\OneDrive\desktop\roi-optimization

Run from project root:
    python src/roi-layer/render_roi_yolo_gif.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _THIS_DIR.parent / "configuration-layer"
_INPUT_DIR = _THIS_DIR.parent / "input-layer"
_YOLO_DIR = _THIS_DIR.parent / "yolo-layer"
_REPO_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_CONFIG_DIR))
sys.path.insert(0, str(_INPUT_DIR))
sys.path.insert(0, str(_YOLO_DIR))

import cv2
import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

from config_node import get_config_value, load_config, validate_config
from detector import build_yolo_layer_package, filter_yolo_detections, initialize_yolo_layer, run_yolo_detection
from input_layer import InputLayer
from roi_layer import apply_roi_to_frame, build_roi_layer_package, initialize_roi_layer, update_roi_state

DEFAULT_OUTPUT_DIR = Path(r"E:\OneDrive\desktop\roi-optimization")
HUD_BG = (24, 24, 24)
HUD_TEXT = (245, 245, 245)
HUD_MUTED = (185, 185, 185)
HUD_ACCENT = (255, 180, 60)
HUD_ROI = (0, 220, 120)
HUD_BOX = (0, 180, 255)


def _resolve_repo_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())


def load_runtime_settings() -> dict[str, object]:
    config_path = _CONFIG_DIR / "config.yaml"
    config = load_config(config_path)
    validate_config(config)
    return {
        "video": _resolve_repo_path(get_config_value(config, "config_input_path")),
        "frame_resolution": tuple(get_config_value(config, "config_frame_resolution")),
        "model": get_config_value(config, "config_yolo_model"),
        "conf": get_config_value(config, "config_yolo_confidence_threshold"),
        "device": get_config_value(config, "config_device"),
        "roi_threshold": get_config_value(config, "config_roi_vehicle_count_threshold"),
    }


def package_to_dict(package) -> dict[str, object]:
    return {
        "input_layer_frame_id": package.input_layer_frame_id,
        "input_layer_timestamp": package.input_layer_timestamp,
        "input_layer_image": package.input_layer_image,
        "input_layer_source_type": package.input_layer_source_type,
        "input_layer_resolution": package.input_layer_resolution,
    }


def translate_detections_to_full_frame(
    detections: list[dict[str, object]],
    roi_bounds: tuple[int, int, int, int] | None,
) -> list[dict[str, object]]:
    if roi_bounds is None:
        return detections
    x_offset, y_offset, _, _ = roi_bounds
    translated: list[dict[str, object]] = []
    for detection in detections:
        x1, y1, x2, y2 = detection["yolo_detection_bbox"]
        translated.append(
            {
                **detection,
                "yolo_detection_bbox": [
                    float(x1) + x_offset,
                    float(y1) + y_offset,
                    float(x2) + x_offset,
                    float(y2) + y_offset,
                ],
            }
        )
    return translated


def draw_text(frame: np.ndarray, text: str, x: int, y: int, *, color: tuple[int, int, int], scale: float = 0.50) -> None:
    cv2.putText(frame, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, detections: list[dict[str, object]], color: tuple[int, int, int]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = [int(v) for v in detection["yolo_detection_bbox"]]
        label = f"{detection['yolo_detection_class']} {float(detection['yolo_detection_confidence']):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_text(frame, label, x1, max(18, y1 - 8), color=color, scale=0.42)


def draw_roi_bounds(frame: np.ndarray, roi_bounds: tuple[int, int, int, int] | None, locked: bool) -> None:
    if roi_bounds is None:
        return
    x1, y1, x2, y2 = [int(v) for v in roi_bounds]
    color = HUD_ROI if locked else HUD_ACCENT
    thickness = 3 if locked else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    draw_text(frame, "ROI LOCKED" if locked else "ROI CALIBRATING", x1 + 4, max(20, y1 - 8), color=color, scale=0.44)


def build_canvas(
    full_frame: np.ndarray,
    *,
    frame_id: int,
    elapsed_seconds: float,
    roi_bounds: tuple[int, int, int, int] | None,
    roi_locked: bool,
    detection_source: str,
    detections: list[dict[str, object]],
    roi_preview: np.ndarray | None,
) -> np.ndarray:
    left = full_frame.copy()
    draw_detections(left, detections, HUD_BOX)
    draw_roi_bounds(left, roi_bounds, roi_locked)

    panel_width = 360
    canvas = np.zeros((left.shape[0], left.shape[1] + panel_width, 3), dtype=np.uint8)
    canvas[:, :left.shape[1]] = left
    canvas[:, left.shape[1]:] = HUD_BG

    panel_x = left.shape[1] + 14
    draw_text(canvas, "ROI -> YOLO GIF RENDER", panel_x, 28, color=HUD_ACCENT, scale=0.62)
    lines = [
        f"Frame: {frame_id}",
        f"Elapsed: {elapsed_seconds:.1f}s",
        f"Stage: {'ROI crop detect' if detection_source == 'roi_crop' else 'ROI calibration'}",
        f"Detection source: {detection_source}",
        f"ROI locked: {roi_locked}",
    ]
    if roi_bounds is not None:
        lines.append(f"ROI bounds: {tuple(int(v) for v in roi_bounds)}")
    y = 70
    for line in lines:
        draw_text(canvas, line, panel_x, y, color=HUD_TEXT, scale=0.48)
        y += 28

    if roi_preview is not None and roi_preview.size > 0:
        preview_h = 220
        preview_w = panel_width - 28
        h, w = roi_preview.shape[:2]
        scale = min(preview_w / max(1, w), preview_h / max(1, h))
        resized = cv2.resize(roi_preview, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        preview_y = canvas.shape[0] - 30 - resized.shape[0]
        preview_x = panel_x + (preview_w - resized.shape[1]) // 2
        canvas[preview_y:preview_y + resized.shape[0], preview_x:preview_x + resized.shape[1]] = resized
        cv2.rectangle(canvas, (preview_x, preview_y), (preview_x + resized.shape[1], preview_y + resized.shape[0]), HUD_ROI, 2)
        draw_text(canvas, "Active ROI crop", preview_x, preview_y - 10, color=HUD_MUTED, scale=0.42)

    return canvas


def save_gif(frames_bgr: list[np.ndarray], output_path: Path, gif_fps: float) -> None:
    if Image is None:
        raise RuntimeError("Pillow is required for GIF export. Install it with: pip install pillow")
    if not frames_bgr:
        raise RuntimeError("No frames collected for GIF export.")
    pil_frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames_bgr]
    duration_ms = max(20, int(round(1000.0 / max(0.1, gif_fps))))
    pil_frames[0].save(
        str(output_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def main() -> None:
    settings = load_runtime_settings()
    parser = argparse.ArgumentParser(description="Render ROI calibration and ROI-cropped YOLO flow into a GIF.")
    parser.add_argument("--video", default=str(settings["video"]), help="Video path to render.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for the exported GIF.")
    parser.add_argument("--max-seconds", type=float, default=12.0, help="Max seconds of processed runtime to render.")
    parser.add_argument("--gif-fps", type=float, default=3.0, help="Playback FPS for exported GIF.")
    parser.add_argument("--max-gif-frames", type=int, default=80, help="Max frames to keep in the GIF.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (_REPO_ROOT / video_path).resolve()

    initialize_yolo_layer(
        model_name=str(settings["model"]),
        conf_threshold=float(settings["conf"]),
        device=str(settings["device"]),
    )
    initialize_roi_layer(
        config_roi_enabled=True,
        config_roi_vehicle_count_threshold=int(settings["roi_threshold"]),
    )

    input_layer = InputLayer()
    input_layer.initialize_input_layer(
        config_input_source="video",
        config_frame_resolution=tuple(settings["frame_resolution"]),
        config_input_path=str(video_path),
    )

    gif_frames: list[np.ndarray] = []
    start_time = time.perf_counter()
    frame_stride = 4
    video_stem = video_path.stem
    output_path = output_dir / f"{video_stem}_roi_pipeline.gif"

    try:
        while True:
            raw_frame = input_layer.read_next_frame()
            if raw_frame is None:
                break

            input_package = input_layer.build_input_layer_package(raw_frame)
            input_pkg = package_to_dict(input_package)

            raw_full = run_yolo_detection(input_pkg)
            filtered_full = filter_yolo_detections(raw_full)
            full_yolo_pkg = build_yolo_layer_package(input_package.input_layer_frame_id, filtered_full)
            roi_state = update_roi_state(input_package, full_yolo_pkg["yolo_layer_detections"])
            roi_locked = bool(roi_state["roi_layer_locked"])
            roi_bounds = roi_state["roi_layer_bounds"]
            roi_preview = apply_roi_to_frame(input_package)
            detection_source = "full_frame_calibration"
            detections = filtered_full

            if roi_locked and roi_bounds is not None:
                roi_pkg = build_roi_layer_package(input_package, roi_preview)
                raw_roi = run_yolo_detection(roi_pkg)
                filtered_roi = filter_yolo_detections(raw_roi)
                detections = translate_detections_to_full_frame(filtered_roi, roi_bounds)
                detection_source = "roi_crop"

            if len(gif_frames) < args.max_gif_frames and (input_package.input_layer_frame_id - 1) % frame_stride == 0:
                gif_frames.append(
                    build_canvas(
                        input_package.input_layer_image,
                        frame_id=input_package.input_layer_frame_id,
                        elapsed_seconds=(time.perf_counter() - start_time),
                        roi_bounds=roi_bounds,
                        roi_locked=roi_locked,
                        detection_source=detection_source,
                        detections=detections,
                        roi_preview=roi_preview,
                    )
                )

            if (time.perf_counter() - start_time) >= float(args.max_seconds):
                break
    finally:
        input_layer.close_input_layer()

    save_gif(gif_frames, output_path, gif_fps=float(args.gif_fps))
    print(f"[roi-render] GIF: {output_path}")


if __name__ == "__main__":
    main()
