#!/usr/bin/env python3
"""
visualize_vlm_frame_cropper.py
Visualize tracked vehicle crop caches on a stable fixed-panel canvas.

Run from src/vlm-frame-cropper-layer:
    python .\visualize_vlm_frame_cropper.py
    python .\visualize_vlm_frame_cropper.py --show

Run from project root:
    python src/vlm-frame-cropper-layer/util/visualize_vlm_frame_cropper.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_CROPPER_DIR = _THIS_DIR.parent
_SRC_DIR = _CROPPER_DIR.parent
_CONFIG_DIR = _SRC_DIR / "configuration-layer"
_INPUT_DIR = _SRC_DIR / "input-layer"
_YOLO_DIR = _SRC_DIR / "yolo-layer"
_TRACKING_DIR = _SRC_DIR / "tracking-layer"
_REPO_ROOT = _SRC_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_CROPPER_DIR))
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
from vlm_frame_cropper_layer import (
    build_vlm_frame_cropper_package,
    build_vlm_frame_cropper_request_package,
    extract_vlm_object_crop,
    initialize_vlm_crop_cache,
    refresh_vlm_crop_cache_track_state,
    update_vlm_crop_cache,
)

STATUS_COLORS = {
    "new": (0, 255, 0),
    "active": (255, 180, 0),
    "lost": (0, 0, 255),
    "dead": (180, 0, 180),
    "no": (170, 170, 170),
    "done": (0, 200, 120),
}
FRAME_TEXT = (245, 245, 245)
SHADOW_TEXT = (0, 0, 0)
PANEL_BG = (18, 18, 18)
CARD_BG = (34, 34, 34)
CELL_BG = (28, 28, 28)
MUTED_TEXT = (180, 180, 180)
SELECTED_BORDER = (255, 255, 255)
# Keep the cropper visualizer at native scale so the preview does not look
# oversized or soft on screen. Users can still resize the OpenCV window.
DISPLAY_SCALE = 1.0
PREVIEW_MAX_WIDTH = 1800
PREVIEW_MAX_HEIGHT = 1000
THUMBNAIL_CACHE = {}


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
        "vlm_crop_cache_size": get_config_value(config, "config_vlm_crop_cache_size"),
        "vlm_dead_after_lost_frames": get_config_value(config, "config_vlm_dead_after_lost_frames"),
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


def tracking_rows(tracking_pkg):
    return [
        {
            "track_id": str(track_id),
            "bbox": tuple(bbox),
            "detector_class": detector_class,
            "confidence": float(confidence),
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


def draw_text_block(frame, lines, origin=(10, 30), color=FRAME_TEXT, scale=0.44, line_gap=22):
    if not lines:
        return
    x, y = origin
    for line in lines:
        cv2.putText(frame, line, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, SHADOW_TEXT, 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += line_gap


def draw_hud_box(frame, lines):
    box_x, box_y = 10, 10
    box_w = 250
    box_h = 28 + (len(lines) * 24)
    hud = frame.copy()
    cv2.rectangle(hud, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(hud, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (70, 70, 70), 1)
    draw_text_block(frame, lines, origin=(box_x + 10, box_y + 22), scale=0.36, line_gap=20)


def draw_tracking_overlay(frame, tracking_pkg):
    for row in tracking_rows(tracking_pkg):
        color = STATUS_COLORS.get(row["status"], (200, 200, 200))
        x1, y1, x2, y2 = [int(value) for value in row["bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {row['track_id']} {row['detector_class']}"
        status_line = f"{row['status']}  conf {row['confidence']:.2f}"
        label_y = max(18, y1 - 8)
        draw_text_block(frame, [label, status_line], origin=(x1 + 4, label_y), color=color, scale=0.28, line_gap=14)


def draw_panel_header(panel, title, subtitle=None):
    draw_text_block(panel, [title], origin=(14, 28), scale=0.66, line_gap=28)
    if subtitle:
        draw_text_block(panel, [subtitle], origin=(14, 58), color=MUTED_TEXT, scale=0.36, line_gap=20)


def paste_with_aspect(panel, image, x, y, target_w, target_h, cache_key=None):
    panel_h, panel_w = panel.shape[:2]
    if x >= panel_w or y >= panel_h or target_w <= 0 or target_h <= 0:
        return
    x2 = min(panel_w, x + target_w)
    y2 = min(panel_h, y + target_h)
    if x2 <= x or y2 <= y:
        return

    cv2.rectangle(panel, (x, y), (x2, y2), CELL_BG, -1)
    if image is None or image.size == 0:
        return

    img_h, img_w = image.shape[:2]
    scale = min((x2 - x) / img_w, (y2 - y) / img_h)
    draw_w = max(1, int(img_w * scale))
    draw_h = max(1, int(img_h * scale))
    if cache_key is not None:
        thumb_key = (cache_key, draw_w, draw_h)
        resized = THUMBNAIL_CACHE.get(thumb_key)
        if resized is None:
            resized = cv2.resize(image, (draw_w, draw_h), interpolation=cv2.INTER_LINEAR)
            THUMBNAIL_CACHE[thumb_key] = resized
    else:
        resized = cv2.resize(image, (draw_w, draw_h), interpolation=cv2.INTER_LINEAR)
    x_offset = x + (target_w - draw_w) // 2
    y_offset = y + (target_h - draw_h) // 2
    paste_w = min(draw_w, panel_w - x_offset)
    paste_h = min(draw_h, panel_h - y_offset)
    if paste_w <= 0 or paste_h <= 0:
        return
    panel[y_offset:y_offset + paste_h, x_offset:x_offset + paste_w] = resized[:paste_h, :paste_w]


def build_cache_rows(cache_state):
    entries = []
    for entry in cache_state["track_caches"].values():
        if entry["cached_crops"]:
            entries.append(entry)
    status_order = {"new": 0, "active": 1, "lost": 2, "dead": 3, "no": 4, "done": 5}
    entries.sort(
        key=lambda entry: (
            status_order.get(entry["vlm_terminal_state"] if entry.get("vlm_terminal_state") in {"dead", "no", "done"} else entry["last_status"], 9),
            -int(entry["last_frame_id"]),
            entry["track_id"],
        )
    )
    return entries


def get_cache_layout(cache_size):
    cache_columns = min(6, max(1, cache_size))
    cache_rows = (max(1, cache_size) + cache_columns - 1) // cache_columns
    return cache_columns, cache_rows


def crop_cache_panel_width_unscaled(cache_size: int) -> int:
    """Horizontal pixel width of the crop-cache panel (matches `content_right` in `build_crop_cache_panel`)."""
    margin_x = 14
    label_w = 116
    thumb_w = 88
    cell_gap = 8
    selected_w = 112
    cache_columns, _ = get_cache_layout(cache_size)
    cache_grid_w = cache_columns * thumb_w + max(0, cache_columns - 1) * cell_gap
    selected_x = margin_x + label_w + cache_grid_w + 18
    return selected_x + selected_w + margin_x


def build_crop_cache_panel(cache_state, frame_height, panel_width):
    cache_size = cache_state["config_vlm_crop_cache_size"]
    panel = np.zeros((frame_height, panel_width, 3), dtype=np.uint8)
    panel[:] = PANEL_BG
    draw_panel_header(
        panel,
        "TRACK CROP CACHE",
        f"Per-track cache size {cache_size}  |  selected column shows VLM candidate",
    )

    rows = build_cache_rows(cache_state)
    if not rows:
        draw_text_block(panel, ["No tracked vehicle crops available yet"], origin=(14, 92), color=MUTED_TEXT, scale=0.42, line_gap=22)
        return panel

    margin_x = 14
    label_w = 116
    thumb_w = 88
    thumb_h = 64
    cell_gap = 8
    cache_row_gap = 18
    selected_w = 112
    selected_h = thumb_h + 14
    header_y = 74
    start_y = 96

    cache_columns, cache_rows = get_cache_layout(cache_size)
    cache_grid_h = cache_rows * thumb_h + max(0, cache_rows - 1) * cache_row_gap
    row_h = max(96, cache_grid_h + 28)

    cache_start_x = margin_x + label_w
    cache_grid_w = cache_columns * thumb_w + max(0, cache_columns - 1) * cell_gap
    selected_x = cache_start_x + cache_grid_w + 18
    content_right = selected_x + selected_w + margin_x
    draw_text_block(panel, ["cache"], origin=(cache_start_x, header_y), color=MUTED_TEXT, scale=0.28, line_gap=14)
    draw_text_block(panel, ["selected"], origin=(selected_x, header_y), color=MUTED_TEXT, scale=0.28, line_gap=14)

    max_rows = max(1, (frame_height - start_y - 16) // row_h)
    clipped_rows = rows[:max_rows]

    for row_index, entry in enumerate(clipped_rows):
        top = start_y + (row_index * row_h)
        bottom = top + row_h - 10
        display_status = entry["vlm_terminal_state"] if entry.get("vlm_terminal_state") in {"dead", "no", "done"} else entry["last_status"]
        row_color = STATUS_COLORS.get(display_status, (220, 220, 220))
        cv2.rectangle(panel, (margin_x, top), (content_right, bottom), CARD_BG, -1)
        cv2.rectangle(panel, (margin_x, top), (content_right, bottom), row_color, 1)

        best_conf = entry["selected_crop"]["confidence"] if entry["selected_crop"] else 0.0
        draw_text_block(
            panel,
            [
                f"ID {entry['track_id']}",
                f"{display_status} {entry['detector_class']}",
                f"best {best_conf:.2f}",
            ],
            origin=(margin_x + 8, top + 18),
            color=row_color,
            scale=0.26,
            line_gap=14,
        )

        cached_crops = entry["cached_crops"]
        for cache_index, crop_info in enumerate(cached_crops):
            cache_col = cache_index % cache_columns
            cache_row = cache_index // cache_columns
            cell_x = cache_start_x + cache_col * (thumb_w + cell_gap)
            cell_y = top + 10 + cache_row * (thumb_h + cache_row_gap)
            cv2.rectangle(panel, (cell_x, cell_y), (cell_x + thumb_w, cell_y + thumb_h), CELL_BG, -1)
            paste_with_aspect(panel, crop_info["crop"], cell_x + 2, cell_y + 2, thumb_w - 4, thumb_h - 4, cache_key=(entry["track_id"], crop_info["frame_id"], "cache"))
            border_color = row_color
            if entry["selected_crop"] is not None and crop_info["frame_id"] == entry["selected_crop"]["frame_id"] and crop_info["confidence"] == entry["selected_crop"]["confidence"]:
                border_color = SELECTED_BORDER
            cv2.rectangle(panel, (cell_x, cell_y), (cell_x + thumb_w, cell_y + thumb_h), border_color, 1)
            draw_text_block(
                panel,
                [f"F{crop_info['frame_id']} {crop_info['confidence']:.2f}"],
                origin=(cell_x, cell_y + thumb_h + 12),
                scale=0.20,
                line_gap=10,
            )

        selected_crop = entry["selected_crop"]
        selected_y = top + max(0, (cache_grid_h - selected_h) // 2)
        cv2.rectangle(panel, (selected_x, selected_y), (selected_x + selected_w, selected_y + selected_h), CELL_BG, -1)
        cv2.rectangle(panel, (selected_x, selected_y), (selected_x + selected_w, selected_y + selected_h), SELECTED_BORDER, 2)
        if selected_crop is not None:
            paste_with_aspect(panel, selected_crop["crop"], selected_x + 4, selected_y + 4, selected_w - 8, thumb_h - 2, cache_key=(entry["track_id"], selected_crop["frame_id"], "selected"))
            draw_text_block(
                panel,
                [
                    f"F{selected_crop['frame_id']} {selected_crop['confidence']:.2f}",
                    selected_crop["trigger_reason"],
                ],
                origin=(selected_x, selected_y + thumb_h + 18),
                scale=0.18,
                line_gap=9,
            )
        else:
            draw_text_block(panel, ["no selection"], origin=(selected_x + 8, selected_y + 40), color=MUTED_TEXT, scale=0.18, line_gap=10)

    if len(rows) > max_rows:
        draw_text_block(panel, [f"Showing {max_rows} of {len(rows)} cached tracks"], origin=(14, frame_height - 18), color=MUTED_TEXT, scale=0.30, line_gap=14)

    return panel


def build_canvas(frame_view, cache_state, panel_width):
    frame_height, frame_width = frame_view.shape[:2]
    display_height = int(frame_height * DISPLAY_SCALE)
    display_width = int(frame_width * DISPLAY_SCALE)
    display_frame = cv2.resize(frame_view, (display_width, display_height), interpolation=cv2.INTER_LINEAR)

    display_panel_width = int(panel_width * DISPLAY_SCALE)
    right_panel_base = build_crop_cache_panel(cache_state, frame_height, panel_width)
    right_panel = cv2.resize(right_panel_base, (display_panel_width, display_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((display_height, display_width + display_panel_width, 3), dtype=np.uint8)
    canvas[:, :display_width] = display_frame
    canvas[:, display_width:] = right_panel
    return canvas


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
    parser.add_argument("--output", default=defaults["output"], help="Optional output video path")
    args = parser.parse_args()

    window_name = "VLM Frame Cropper Visualizer"
    preview_window_sized = False
    if args.show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    frame_resolution = tuple(defaults["frame_resolution"])
    fps, width, height = probe_video_metadata(args.input_source, args.video, frame_resolution)
    panel_width = crop_cache_panel_width_unscaled(defaults["vlm_crop_cache_size"])
    out = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        output_width = int((width + panel_width) * DISPLAY_SCALE)
        output_height = int(height * DISPLAY_SCALE)
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height))
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
    crop_cache_state = initialize_vlm_crop_cache(defaults["vlm_crop_cache_size"], defaults["vlm_dead_after_lost_frames"])

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

            base_frame_view = input_package.input_layer_image.copy()
            elapsed = time.time() - start_time
            fps_text = input_package.input_layer_frame_id / elapsed if elapsed > 0 else 0.0
            cached_track_count = len([entry for entry in crop_cache_state["track_caches"].values() if entry["cached_crops"]])

            for index, row in enumerate(tracking_rows(tracking_pkg)):
                refresh_vlm_crop_cache_track_state(
                    vlm_crop_cache_state=crop_cache_state,
                    tracking_layer_row=row,
                    vlm_frame_cropper_frame_id=input_package.input_layer_frame_id,
                )
                if row["status"] == "lost":
                    continue

                trigger_reason = f"tracking_status:{row['status']}"
                request_pkg = build_vlm_frame_cropper_request_package(
                    input_layer_package=input_package,
                    tracking_layer_package=tracking_pkg,
                    track_index=index,
                    vlm_frame_cropper_trigger_reason=trigger_reason,
                    config_vlm_enabled=True,
                )
                if request_pkg is None:
                    continue
                crop = extract_vlm_object_crop(input_package, request_pkg)
                crop_pkg = build_vlm_frame_cropper_package(request_pkg, crop)
                update_vlm_crop_cache(
                    vlm_crop_cache_state=crop_cache_state,
                    tracking_layer_row=row,
                    vlm_frame_cropper_layer_package=crop_pkg,
                    vlm_frame_cropper_frame_id=input_package.input_layer_frame_id,
                    vlm_frame_cropper_trigger_reason=trigger_reason,
                )

            canvas = build_canvas(base_frame_view, crop_cache_state, panel_width)
            display_tracking_pkg = {key: value for key, value in tracking_pkg.items()}
            for index, bbox in enumerate(display_tracking_pkg["tracking_layer_bbox"]):
                display_tracking_pkg["tracking_layer_bbox"][index] = [int(coord * DISPLAY_SCALE) for coord in bbox]
            left_width = int(base_frame_view.shape[1] * DISPLAY_SCALE)
            draw_tracking_overlay(canvas[:, :left_width], display_tracking_pkg)
            draw_hud_box(
                canvas[:, :left_width],
                [
                    f"Frame: {input_package.input_layer_frame_id}",
                    f"Tracks: {len(tracking_pkg['tracking_layer_track_id'])}",
                    f"Cached tracks: {cached_track_count}",
                    f"Cache size: {defaults['vlm_crop_cache_size']}",
                    f"VLM flag: {defaults['vlm_enabled']} (info)",
                    f"FPS: {fps_text:.1f}",
                ],
            )
            if out is not None:
                out.write(canvas)

            if args.show:
                cv2.imshow(window_name, canvas)
                if not preview_window_sized:
                    canvas_height, canvas_width = canvas.shape[:2]
                    preview_width = min(canvas_width, PREVIEW_MAX_WIDTH)
                    preview_height = min(canvas_height, PREVIEW_MAX_HEIGHT)
                    cv2.resizeWindow(window_name, preview_width, preview_height)
                    preview_window_sized = True
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
        print(f"Saved VLM frame cropper visualization to: {args.output}")


if __name__ == "__main__":
    main()
