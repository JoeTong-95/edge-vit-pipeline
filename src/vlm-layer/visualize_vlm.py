#!/usr/bin/env python3
"""
visualize_vlm.py
Show the actual cropper -> VLM -> ack -> metadata loop for one focus track at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_CONFIG_DIR = _THIS_DIR.parent / "configuration-layer"
_INPUT_DIR = _THIS_DIR.parent / "input-layer"
_YOLO_DIR = _THIS_DIR.parent / "yolo-layer"
_TRACKING_DIR = _THIS_DIR.parent / "tracking-layer"
_CROPPER_LAYER_DIR = _THIS_DIR.parent / "vlm-frame-cropper-layer"
_VEHICLE_STATE_DIR = _THIS_DIR.parent / "vehicle-state-layer"

for path in (_THIS_DIR, _CONFIG_DIR, _INPUT_DIR, _YOLO_DIR, _TRACKING_DIR, _CROPPER_LAYER_DIR, _VEHICLE_STATE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import cv2
import numpy as np
from config_node import get_config_value, load_config, validate_config
from detector import build_yolo_layer_package, filter_yolo_detections, initialize_yolo_layer, run_yolo_detection
from input_layer import InputLayer
from layer import (
    DEFAULT_VLM_QUERY_TYPE,
    VLMConfig,
    VLMFrameCropperLayerPackage,
    build_vlm_ack_package,
    build_vlm_ack_package_from_result,
    build_vlm_layer_package,
    initialize_vlm_layer,
    normalize_vlm_result,
    prepare_vlm_prompt,
    run_vlm_inference,
)
from tracker import assign_tracking_status, build_tracking_layer_package, initialize_tracking_layer, update_tracks
from vehicle_state_layer import (
    get_vehicle_state_record,
    initialize_vehicle_state_layer,
    update_vehicle_state_from_tracking,
    update_vehicle_state_from_vlm,
    update_vehicle_state_from_vlm_ack,
)
from vlm_frame_cropper_layer import (
    build_vlm_dispatch_package,
    build_vlm_frame_cropper_package,
    build_vlm_frame_cropper_request_package,
    extract_vlm_object_crop,
    initialize_vlm_crop_cache,
    refresh_vlm_crop_cache_track_state,
    register_vlm_ack_package,
    update_vlm_crop_cache,
)

DISPLAY_SCALE = 1.45
LEFT_BG = (18, 18, 20)
RIGHT_BG = (16, 16, 20)
CARD_BG = (32, 32, 38)
MUTED = (170, 170, 180)
TEXT = (245, 245, 248)
ACCENT = (80, 210, 255)
GOOD = (70, 210, 110)
WARN = (0, 190, 255)
BAD = (70, 90, 255)
STATUS_COLORS = {"new": (60, 230, 90), "active": (255, 180, 0), "lost": (40, 80, 255)}
MAX_JSON_CHARS = 440


def _resolve_repo_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())


def _resolve_vlm_model_path(config_value: str) -> str:
    raw = (config_value or "").strip()
    path = Path(raw)
    if path.is_absolute() and path.exists():
        return str(path.resolve())
    candidate = _REPO_ROOT / raw
    if candidate.exists():
        return str(candidate.resolve())
    candidate = _THIS_DIR / raw
    if candidate.exists():
        return str(candidate.resolve())
    return str((_REPO_ROOT / raw).resolve())


def load_runtime_settings() -> dict[str, Any]:
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
        "vlm_model": _resolve_vlm_model_path(get_config_value(config, "config_vlm_model")),
        "vlm_crop_feedback_enabled": get_config_value(config, "config_vlm_crop_feedback_enabled"),
        "vlm_crop_cache_size": get_config_value(config, "config_vlm_crop_cache_size"),
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


def draw_text(frame, text, x, y, color=TEXT, scale=0.42, thickness=1):
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), max(2, thickness + 1), cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_text_block(frame, lines, origin, color=TEXT, scale=0.36, line_gap=18):
    x, y = origin
    for line in lines:
        draw_text(frame, line, x, y, color=color, scale=scale, thickness=1)
        y += line_gap


def wrap_text(text: str, max_chars: int) -> list[str]:
    if not text:
        return [""]
    out = []
    for raw in text.splitlines():
        chunk = raw.strip()
        if not chunk:
            out.append("")
            continue
        while chunk:
            if len(chunk) <= max_chars:
                out.append(chunk)
                break
            cut = chunk.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(chunk[:cut].rstrip())
            chunk = chunk[cut:].lstrip()
    return out


def paste_with_aspect(canvas, image, x, y, w, h):
    canvas_h, canvas_w = canvas.shape[:2]
    if x >= canvas_w or y >= canvas_h or w <= 0 or h <= 0:
        return
    x2 = min(canvas_w, x + w)
    y2 = min(canvas_h, y + h)
    if x2 <= x or y2 <= y:
        return
    cv2.rectangle(canvas, (x, y), (x2, y2), CARD_BG, -1)
    if image is None or getattr(image, 'size', 0) == 0:
        if y + h // 2 < canvas_h:
            draw_text(canvas, 'none', x + 10, min(canvas_h - 4, y + h // 2), color=MUTED, scale=0.42)
        cv2.rectangle(canvas, (x, y), (x2, y2), (90, 90, 100), 1)
        return
    ih, iw = image.shape[:2]
    scale = min(max(1, x2 - x - 8) / iw, max(1, y2 - y - 8) / ih)
    dw = max(1, int(iw * scale))
    dh = max(1, int(ih * scale))
    resized = cv2.resize(image, (dw, dh), interpolation=cv2.INTER_LINEAR)
    ox = x + max(0, (x2 - x - dw) // 2)
    oy = y + max(0, (y2 - y - dh) // 2)
    paste_w = min(dw, canvas_w - ox)
    paste_h = min(dh, canvas_h - oy)
    if paste_w > 0 and paste_h > 0:
        canvas[oy:oy + paste_h, ox:ox + paste_w] = resized[:paste_h, :paste_w]
    cv2.rectangle(canvas, (x, y), (x2, y2), (120, 120, 130), 1)


def pick_focus_track(crop_cache_state: dict[str, Any], dispatch_track_ids: list[str]) -> str | None:
    if dispatch_track_ids:
        return dispatch_track_ids[-1]
    entries = list(crop_cache_state['track_caches'].values())
    if not entries:
        return None

    def rank(entry):
        priority = 0
        if entry['vlm_request_in_flight']:
            priority = 5
        elif entry['vlm_retry_requested']:
            priority = 4
        elif entry['vlm_previous_sent_must_be_used']:
            priority = 3
        elif entry['vlm_last_sent_package'] is not None:
            priority = 2
        elif entry['selected_crop'] is not None:
            priority = 1
        return (priority, int(entry['last_frame_id']))

    best = max(entries, key=rank)
    return best['track_id'] if rank(best)[0] > 0 else None


def derive_loop_state(track_cache: dict[str, Any], vehicle_record: dict[str, Any] | None) -> tuple[str, tuple[int, int, int], list[str]]:
    cache_count = len(track_cache['cached_crops'])
    if vehicle_record and vehicle_record.get('vehicle_state_layer_vlm_called'):
        return (
            'PROGRESSED TO METADATA',
            GOOD,
            [
                'This track already has an accepted VLM result.',
                'Tracking can continue, but cropper and VLM are skipped for this truck.',
            ],
        )
    if track_cache['vlm_previous_sent_must_be_used']:
        return (
            'LOST AFTER RETRY',
            BAD,
            [
                'VLM asked for a better image.',
                'The object left scope before a new round filled.',
                'VLM must work with the previously sent image.',
            ],
        )
    if track_cache['vlm_request_in_flight']:
        return (
            'WAITING FOR VLM',
            WARN,
            ['Selected image has been sent to VLM.', 'Waiting for accepted or retry_requested.'],
        )
    if track_cache['vlm_finalized'] and track_cache['vlm_ack_status'] == 'accepted':
        return (
            'ACCEPTED',
            GOOD,
            ['VLM accepted the image as good enough.', 'Structured JSON can now feed metadata output.'],
        )
    if track_cache['vlm_retry_requested']:
        return (
            'REFILLING CACHE',
            WARN,
            ['VLM requested a better image.', 'Cropper cleared the old round and is collecting a new one.'],
        )
    return (
        'COLLECTING FIRST ROUND',
        ACCENT,
        [f'Collecting current round until cache is full ({cache_count}).', 'Then cropper scores that round and sends one best image.'],
    )


def pretty_json_snippet(data: dict[str, Any]) -> list[str]:
    text = json.dumps(data, indent=2)
    if len(text) > MAX_JSON_CHARS:
        text = text[:MAX_JSON_CHARS] + '...'
    return text.splitlines()


def build_right_panel(panel_h: int, panel_w: int, focus_track_id: str | None, crop_cache_state: dict[str, Any], debug_state: dict[str, Any], vehicle_record: dict[str, Any] | None, cache_size: int, feedback_enabled: bool) -> np.ndarray:
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = RIGHT_BG
    draw_text(panel, 'VLM LOOP VIEW', 16, 28, scale=0.78)
    if focus_track_id is None:
        draw_text_block(panel, ['No eligible focus track yet.', 'Need a tracked object and crop cache activity.'], (18, 70), color=MUTED, scale=0.42, line_gap=24)
        return panel

    track_cache = crop_cache_state['track_caches'][focus_track_id]
    debug = debug_state.get(focus_track_id, {})
    loop_title, loop_color, loop_lines = derive_loop_state(track_cache, vehicle_record)

    draw_text(panel, f'Focus track {focus_track_id}', 18, 64, color=ACCENT, scale=0.52)
    draw_text(panel, loop_title, 18, 92, color=loop_color, scale=0.58)
    draw_text(panel, f'feedback_loop={feedback_enabled}', 18, 110, color=MUTED, scale=0.28)
    draw_text_block(panel, [f'class={track_cache["detector_class"]}', f'cache {len(track_cache["cached_crops"])} / {cache_size}', f'sent={track_cache["vlm_sent_count"]}'], (18, 120), color=MUTED, scale=0.34, line_gap=18)

    ack_status = debug.get('ack_status', track_cache.get('vlm_ack_status', 'not_requested'))
    reasons = debug.get('retry_reasons', [])
    note = debug.get('image_quality_notes', '') or track_cache.get('vlm_ack_reason', '')
    summary_y = 176
    cv2.rectangle(panel, (14, summary_y), (panel_w - 14, summary_y + 72), CARD_BG, -1)
    draw_text(panel, 'Latest VLM reply', 22, summary_y + 22, scale=0.40)
    if ack_status == 'retry_requested':
        reason_text = ', '.join(reasons) if reasons else (note or 'occluded')
        draw_text_block(panel, [f'ack=retry_requested', f'reasons: {reason_text}', note or 'VLM needs a better image.'], (24, summary_y + 46), color=WARN, scale=0.31, line_gap=16)
    elif debug.get('normalized_result') is not None:
        draw_text_block(panel, [f'ack=accepted', 'JSON is clear enough for metadata output.'], (24, summary_y + 46), color=GOOD, scale=0.31, line_gap=16)
    else:
        draw_text_block(panel, ['No VLM reply yet for this track.'], (24, summary_y + 46), color=MUTED, scale=0.31, line_gap=16)

    card_y = 268
    card_w = (panel_w - 54) // 2
    card_h = 188
    selected = track_cache['selected_crop']
    last_sent = track_cache['vlm_last_sent_package']
    draw_text(panel, 'Selected candidate', 18, card_y - 8, scale=0.40)
    draw_text(panel, 'Actual image sent to VLM', 36 + card_w, card_y - 8, scale=0.40)
    paste_with_aspect(panel, selected['crop'] if selected else None, 18, card_y, card_w, card_h)
    paste_with_aspect(panel, last_sent['crop'] if last_sent else None, 36 + card_w, card_y, card_w, card_h)

    selected_lines = ['not ready']
    if selected:
        selected_lines = [f'frame {selected["frame_id"]}  conf {selected["confidence"]:.2f}', f'score {selected["selection_key"]:.3f}', selected['trigger_reason']]
    sent_lines = ['nothing sent yet']
    if last_sent:
        sent_lines = [f'frame {last_sent["frame_id"]}  conf {last_sent["confidence"]:.2f}', track_cache['vlm_last_dispatch_mode'] or 'dispatch pending', track_cache['vlm_last_dispatch_reason'] or '']
    draw_text_block(panel, selected_lines, (20, card_y + card_h + 18), scale=0.31, line_gap=16)
    draw_text_block(panel, sent_lines, (38 + card_w, card_y + card_h + 18), scale=0.31, line_gap=16)

    strip_y = card_y + card_h + 82
    draw_text(panel, 'Current cache round', 18, strip_y - 8, scale=0.40)
    thumbs = track_cache['cached_crops'][-min(6, cache_size):]
    thumb_w, thumb_h, gap = 84, 60, 8
    for idx, crop in enumerate(thumbs):
        tx = 18 + idx * (thumb_w + gap)
        paste_with_aspect(panel, crop['crop'], tx, strip_y, thumb_w, thumb_h)
        border = GOOD if selected and crop['frame_id'] == selected['frame_id'] and crop['confidence'] == selected['confidence'] else STATUS_COLORS.get(crop['status'], MUTED)
        cv2.rectangle(panel, (tx, strip_y), (tx + thumb_w, strip_y + thumb_h), border, 1)
        draw_text(panel, f'F{crop["frame_id"]}', tx, strip_y + thumb_h + 14, scale=0.24)

    loop_box_y = strip_y + thumb_h + 28
    cv2.rectangle(panel, (14, loop_box_y), (panel_w - 14, loop_box_y + 84), CARD_BG, -1)
    draw_text(panel, 'How the loop behaves now', 22, loop_box_y + 22, scale=0.40)
    draw_text_block(panel, loop_lines, (24, loop_box_y + 46), scale=0.31, line_gap=16)

    judge_y = loop_box_y + 98
    cv2.rectangle(panel, (14, judge_y), (panel_w - 14, judge_y + 132), CARD_BG, -1)
    draw_text(panel, 'What VLM returned', 22, judge_y + 22, scale=0.40)
    normalized = debug.get('normalized_result')
    if ack_status == 'retry_requested':
        reason_text = ', '.join(reasons) if reasons else track_cache.get('vlm_ack_reason', 'occluded')
        judge_lines = [f'ack=retry_requested', f'reasons: {reason_text}', debug.get('image_quality_notes', '') or 'VLM wants a better image.']
        draw_text_block(panel, judge_lines, (24, judge_y + 50), color=WARN, scale=0.31, line_gap=16)
    elif normalized:
        draw_text_block(panel, ['ack=accepted', 'JSON classification:'], (24, judge_y + 46), color=GOOD, scale=0.31, line_gap=16)
        json_lines = pretty_json_snippet({
            'is_truck': normalized.get('is_truck'),
            'truck_type': normalized.get('truck_type'),
            'wheel_count': normalized.get('wheel_count'),
            'estimated_weight_kg': normalized.get('estimated_weight_kg'),
            'confidence': normalized.get('vlm_layer_confidence'),
        })
        draw_text_block(panel, json_lines, (28, judge_y + 82), scale=0.26, line_gap=14)
    else:
        draw_text_block(panel, ['No VLM response yet.'], (24, judge_y + 52), color=MUTED, scale=0.31, line_gap=16)

    meta_y = judge_y + 156
    cv2.rectangle(panel, (14, meta_y), (panel_w - 14, panel_h - 14), CARD_BG, -1)
    draw_text(panel, 'Metadata output', 22, meta_y + 22, scale=0.40)
    if vehicle_record and vehicle_record.get('vehicle_state_layer_vlm_called'):
        tags = vehicle_record.get('vehicle_state_layer_semantic_tags', {})
        tag_text = ', '.join(f'{k}={v}' for k, v in list(tags.items())[:4]) or 'none'
        meta_lines = [
            f"is_truck={vehicle_record.get('vehicle_state_layer_semantic_tags', {}).get('is_truck', 'unknown')}",
            f"truck_type={vehicle_record.get('vehicle_state_layer_truck_type', 'unknown')}",
            f"vlm_called={vehicle_record.get('vehicle_state_layer_vlm_called')}",
            f"ack_status={vehicle_record.get('vehicle_state_layer_vlm_ack_status')}",
            f'tags: {tag_text}',
        ]
        draw_text_block(panel, meta_lines, (24, meta_y + 48), scale=0.31, line_gap=16)
    else:
        draw_text_block(panel, ['Metadata has not been emitted for this track yet.', 'That only happens after VLM accepts the image.'], (24, meta_y + 48), color=MUTED, scale=0.31, line_gap=18)
    return panel


def build_canvas(frame_view: np.ndarray, tracking_pkg: dict[str, Any], focus_track_id: str | None, crop_cache_state: dict[str, Any], debug_state: dict[str, Any], vehicle_record: dict[str, Any] | None, cache_size: int, fps_text: float, frame_id: int, feedback_enabled: bool) -> np.ndarray:
    raw_h, raw_w = frame_view.shape[:2]
    disp_w = int(raw_w * DISPLAY_SCALE)
    disp_h = int(raw_h * DISPLAY_SCALE)
    right_w = 780
    left = cv2.resize(frame_view, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
    for row in tracking_rows(tracking_pkg):
        x1, y1, x2, y2 = [int(v * DISPLAY_SCALE) for v in row['bbox']]
        state_record = get_vehicle_state_record(row['track_id'])
        progressed_only_tracking = bool(state_record and state_record.get('vehicle_state_layer_vlm_called'))
        display_status = 'done' if progressed_only_tracking else row['status']
        color = GOOD if progressed_only_tracking else STATUS_COLORS.get(row['status'], (220, 220, 220))
        thickness = 2
        if focus_track_id is not None and row['track_id'] == focus_track_id and not progressed_only_tracking:
            color = ACCENT
            thickness = 3
        cv2.rectangle(left, (x1, y1), (x2, y2), color, thickness)
        draw_text(left, f"ID {row['track_id']} {row['detector_class']} {display_status}", x1 + 4, max(18, y1 - 8), color=color, scale=0.34)

    overlay = left.copy()
    cv2.rectangle(overlay, (10, 10), (240, 108), RIGHT_BG, -1)
    cv2.addWeighted(overlay, 0.65, left, 0.35, 0, left)
    draw_text_block(left, [f'Frame: {frame_id}', f'Tracks: {len(tracking_pkg["tracking_layer_track_id"])}', f'Focus: {focus_track_id or "none"}', f'FPS: {fps_text:.1f}'], (18, 30), scale=0.36, line_gap=20)

    right = build_right_panel(disp_h, right_w, focus_track_id, crop_cache_state, debug_state, vehicle_record, cache_size, feedback_enabled)
    return np.hstack([left, right])


def main() -> None:
    defaults = load_runtime_settings()
    parser = argparse.ArgumentParser(description='Visualize the cropper -> VLM -> ack -> metadata loop')
    parser.add_argument('--input-source', default=defaults['input_source'], choices=['video', 'camera'])
    parser.add_argument('--video', default=defaults['video'])
    parser.add_argument('--model', default=defaults['model'])
    parser.add_argument('--conf', type=float, default=defaults['conf'])
    parser.add_argument('--device', default=defaults['device'])
    parser.add_argument('--vlm-device', default='', help='Override device for VLM only')
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--gstreamer', action='store_true')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--output', default=defaults['output'], help='Optional output video path')
    args = parser.parse_args()

    frame_resolution = tuple(defaults['frame_resolution'])
    fps, width, height = probe_video_metadata(args.input_source, args.video, frame_resolution)
    out = None
    output_w = int(width * DISPLAY_SCALE) + 780
    output_h = int(height * DISPLAY_SCALE)
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_w, output_h))
        if not out.isOpened():
            raise RuntimeError(f'Could not create output video: {args.output}')

    vlm_device = args.vlm_device.strip() or args.device
    vlm_state = initialize_vlm_layer(VLMConfig(config_vlm_enabled=bool(defaults['vlm_enabled']), config_vlm_model=defaults['vlm_model'], config_device=vlm_device))
    input_layer = InputLayer()
    input_layer.initialize_input_layer(config_input_source=args.input_source, config_frame_resolution=frame_resolution, config_input_path=args.video, camera_device_index=args.camera_index, use_gstreamer=args.gstreamer)
    initialize_yolo_layer(model_name=args.model, conf_threshold=args.conf, device=args.device)
    initialize_tracking_layer(frame_rate=int(fps))
    crop_cache_state = initialize_vlm_crop_cache(defaults['vlm_crop_cache_size'])
    initialize_vehicle_state_layer(prune_after_lost_frames=None)
    debug_state: dict[str, dict[str, Any]] = {}
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
            update_vehicle_state_from_tracking(tracking_pkg)

            frame_view = input_package.input_layer_image.copy()
            dispatch_track_ids: list[str] = []
            for index, row in enumerate(tracking_rows(tracking_pkg)):
                refresh_vlm_crop_cache_track_state(crop_cache_state, row, input_package.input_layer_frame_id)
                vehicle_record_for_row = get_vehicle_state_record(row['track_id'])
                if vehicle_record_for_row and vehicle_record_for_row.get('vehicle_state_layer_vlm_called'):
                    debug_state.setdefault(str(row['track_id']), {})['progressed_only_tracking'] = True
                    continue
                if row['status'] == 'lost':
                    continue
                request_pkg = build_vlm_frame_cropper_request_package(input_layer_package=input_package, tracking_layer_package=tracking_pkg, track_index=index, vlm_frame_cropper_trigger_reason=f'tracking_status:{row["status"]}', config_vlm_enabled=True)
                if request_pkg is None:
                    continue
                crop = extract_vlm_object_crop(input_package, request_pkg)
                crop_pkg = build_vlm_frame_cropper_package(request_pkg, crop)
                update_vlm_crop_cache(crop_cache_state, row, crop_pkg, input_package.input_layer_frame_id, f'tracking_status:{row["status"]}')

            if vlm_state.config_vlm_enabled:
                for tid in sorted(crop_cache_state['track_caches'].keys(), key=lambda x: (len(str(x)), str(x))):
                    vehicle_record_for_dispatch = get_vehicle_state_record(tid)
                    if vehicle_record_for_dispatch and vehicle_record_for_dispatch.get('vehicle_state_layer_vlm_called'):
                        continue
                    dispatch = build_vlm_dispatch_package(crop_cache_state, tid)
                    if dispatch is None:
                        continue
                    dispatch_track_ids.append(str(tid))
                    inner = dispatch['vlm_frame_cropper_layer_package']
                    vlm_crop_pkg = VLMFrameCropperLayerPackage(vlm_frame_cropper_layer_track_id=str(inner['vlm_frame_cropper_layer_track_id']), vlm_frame_cropper_layer_image=inner['vlm_frame_cropper_layer_image'], vlm_frame_cropper_layer_bbox=tuple(int(x) for x in inner['vlm_frame_cropper_layer_bbox']))
                    prompt_text = prepare_vlm_prompt(DEFAULT_VLM_QUERY_TYPE, vlm_crop_pkg)
                    try:
                        query_type = DEFAULT_VLM_QUERY_TYPE if defaults['vlm_crop_feedback_enabled'] else 'vehicle_semantics_single_shot_v1'
                        raw_result = run_vlm_inference(vlm_state, vlm_crop_pkg, query_type)
                        normalized = normalize_vlm_result(raw_result)
                        ack = build_vlm_ack_package_from_result(raw_result) if defaults['vlm_crop_feedback_enabled'] else build_vlm_ack_package(str(tid), 'accepted', 'feedback_disabled_single_shot', False)
                        register_vlm_ack_package(crop_cache_state, ack)
                        update_vehicle_state_from_vlm_ack(ack)
                        if ack.vlm_ack_status == 'accepted':
                            update_vehicle_state_from_vlm(build_vlm_layer_package(raw_result))
                        debug_state[str(tid)] = {
                            'prompt': prompt_text,
                            'raw_response': raw_result.vlm_layer_raw_text.strip(),
                            'normalized_result': normalized,
                            'ack_status': ack.vlm_ack_status,
                            'ack_reason': ack.vlm_ack_reason,
                            'retry_reasons': normalized.get('vlm_retry_reasons', []) if defaults['vlm_crop_feedback_enabled'] else [],
                            'image_quality_notes': normalized.get('vlm_image_quality_notes', '') if defaults['vlm_crop_feedback_enabled'] else 'feedback loop disabled; first dispatch is final',
                            'progressed_only_tracking': ack.vlm_ack_status == 'accepted',
                        }
                    except Exception as exc:
                        ack = build_vlm_ack_package(str(tid), 'retry_requested', f'VLM error: {exc}', True)
                        register_vlm_ack_package(crop_cache_state, ack)
                        update_vehicle_state_from_vlm_ack(ack)
                        debug_state[str(tid)] = {
                            'prompt': prompt_text,
                            'raw_response': f'ERROR: {exc}',
                            'normalized_result': None,
                            'ack_status': 'retry_requested',
                            'ack_reason': f'VLM error: {exc}',
                            'retry_reasons': ['occluded'],
                            'image_quality_notes': f'VLM error: {exc}',
                        }

            focus_track_id = pick_focus_track(crop_cache_state, dispatch_track_ids)
            vehicle_record = get_vehicle_state_record(focus_track_id) if focus_track_id is not None else None
            elapsed = time.time() - start_time
            fps_text = input_package.input_layer_frame_id / elapsed if elapsed > 0 else 0.0
            canvas = build_canvas(frame_view, tracking_pkg, focus_track_id, crop_cache_state, debug_state, vehicle_record, defaults['vlm_crop_cache_size'], fps_text, input_package.input_layer_frame_id, bool(defaults['vlm_crop_feedback_enabled']))
            if out is not None:
                out.write(canvas)
            if args.show:
                cv2.imshow('VLM loop visualizer', canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if args.max_frames > 0 and input_package.input_layer_frame_id >= args.max_frames:
                break
    finally:
        input_layer.close_input_layer()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

    if args.output:
        print(f'Saved VLM visualization to: {args.output}')


if __name__ == '__main__':
    main()
