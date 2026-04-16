#!/usr/bin/env python3
"""
visualize_roi_vlm_upson.py

Clip-specific ROI + tracking + cropper + VLM visualizer for the Upson video.

- Starts at 0:48
- Stops at 1:48
- Paces display/output to the source FPS by default
- Keeps the richer VLM status coloring from the main ROI VLM visualizer path
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_ROI_DIR = _THIS_DIR.parent
_SRC_DIR = _ROI_DIR.parent
_REPO_ROOT = _SRC_DIR.parent
_CONFIG_DIR = _SRC_DIR / "configuration-layer"
_INPUT_DIR = _SRC_DIR / "input-layer"
_YOLO_DIR = _SRC_DIR / "yolo-layer"
_TRACKING_DIR = _SRC_DIR / "tracking-layer"
_CROPPER_LAYER_DIR = _SRC_DIR / "vlm-frame-cropper-layer"
_VEHICLE_STATE_DIR = _SRC_DIR / "vehicle-state-layer"
_VLM_DIR = _SRC_DIR / "vlm-layer"
_VLM_UTIL_DIR = _VLM_DIR / "util"

for path in (
    _THIS_DIR,
    _ROI_DIR,
    _CONFIG_DIR,
    _INPUT_DIR,
    _YOLO_DIR,
    _TRACKING_DIR,
    _CROPPER_LAYER_DIR,
    _VEHICLE_STATE_DIR,
    _VLM_DIR,
    _VLM_UTIL_DIR,
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import cv2

from config_node import get_config_value, load_config, validate_config
from detector import build_yolo_layer_package, filter_yolo_detections, initialize_yolo_layer, run_yolo_detection
from input_layer import InputLayer
from layer import (
    DEFAULT_VLM_QUERY_TYPE,
    VLMConfig,
    VLMFrameCropperLayerPackage,
    build_vlm_layer_package,
    initialize_vlm_layer,
    prepare_vlm_prompt,
)
from roi_layer import apply_roi_to_frame, build_roi_layer_package, initialize_roi_layer, update_roi_state
from tracker import assign_tracking_status, build_tracking_layer_package, initialize_tracking_layer, update_tracks
from vehicle_state_layer import (
    get_vehicle_state_record,
    initialize_vehicle_state_layer,
    update_vehicle_state_from_tracking,
    update_vehicle_state_from_vlm,
    update_vehicle_state_from_vlm_ack,
)
from visualize_vlm import (
    DISPLAY_SCALE,
    build_canvas,
    draw_text,
    draw_text_block,
    input_package_to_dict,
    pick_focus_track,
    probe_video_metadata,
    tracking_rows,
)
from visualize_vlm_realtime import AsyncVLMWorker, overlay_async_status
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

ROI_COLOR = (0, 200, 255)
LOCKED_COLOR = (0, 220, 120)
PANEL_BG = (20, 20, 24)

DEFAULT_START_SECONDS = 48.0
DEFAULT_END_SECONDS = 108.0


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
    candidate = _VLM_DIR / raw
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
        "vlm_dead_after_lost_frames": get_config_value(config, "config_vlm_dead_after_lost_frames"),
        "roi_enabled": get_config_value(config, "config_roi_enabled"),
        "roi_threshold": get_config_value(config, "config_roi_vehicle_count_threshold"),
        "output": "",
    }


def maybe_sleep_to_source_fps(start_wall: float, clip_frame_index: int, fps: float) -> None:
    if fps <= 0:
        return
    target_elapsed = clip_frame_index / fps
    sleep_sec = target_elapsed - (time.time() - start_wall)
    if sleep_sec > 0:
        time.sleep(sleep_sec)


def translate_tracking_pkg_to_full_frame(
    tracking_pkg: dict[str, Any],
    roi_bounds: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    if roi_bounds is None:
        return tracking_pkg
    x1, y1, _, _ = [int(v) for v in roi_bounds]
    translated = dict(tracking_pkg)
    translated["tracking_layer_bbox"] = [
        [int(bx1) + x1, int(by1) + y1, int(bx2) + x1, int(by2) + y1]
        for bx1, by1, bx2, by2 in tracking_pkg["tracking_layer_bbox"]
    ]
    return translated


def build_calibration_canvas(
    frame_view,
    calibration_detections,
    roi_state,
    roi_pkg,
    fps_text,
    frame_id,
    clip_frame_index,
    target_count,
):
    left = frame_view.copy()
    for detection in calibration_detections:
        x1, y1, x2, y2 = [int(value) for value in detection["yolo_detection_bbox"]]
        cv2.rectangle(left, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if roi_state["roi_layer_bounds"] is not None:
        x1, y1, x2, y2 = [int(value) for value in roi_state["roi_layer_bounds"]]
        color = LOCKED_COLOR if roi_state["roi_layer_locked"] else ROI_COLOR
        cv2.rectangle(left, (x1, y1), (x2, y2), color, 3 if roi_state["roi_layer_locked"] else 2)
        draw_text_block(left, ["ROI LOCKED" if roi_state["roi_layer_locked"] else "ROI CALIBRATING"], (x1 + 6, max(22, y1 - 10)), color=color, scale=0.30, line_gap=16)
    overlay = left.copy()
    cv2.rectangle(overlay, (10, 10), (380, 152), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.65, left, 0.35, 0, left)
    draw_text_block(
        left,
        [
            f"Clip frame: {clip_frame_index}",
            f"Source frame: {frame_id}",
            "Phase: ROI calibration",
            f"Threshold: {target_count}",
            f"Candidate count: {roi_state['roi_candidate_box_count']}",
            f"Current detections: {len(calibration_detections)}",
            f"Feed FPS: {fps_text:.1f}",
        ],
        (18, 30),
        scale=0.36,
        line_gap=20,
    )

    raw_h, raw_w = left.shape[:2]
    disp_w = int(raw_w * DISPLAY_SCALE)
    disp_h = int(raw_h * DISPLAY_SCALE)
    right_w = 780
    left = cv2.resize(left, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
    panel = cv2.resize(roi_pkg["roi_layer_image"], (right_w, disp_h), interpolation=cv2.INTER_LINEAR)
    overlay_panel = panel.copy()
    cv2.rectangle(overlay_panel, (0, 0), (right_w, disp_h), PANEL_BG, -1)
    cv2.addWeighted(overlay_panel, 0.55, panel, 0.45, 0, panel)
    draw_text(panel, "ROI CROP PREVIEW", 16, 28, scale=0.70)
    draw_text_block(
        panel,
        [
            "Full-frame YOLO is only used here for ROI lock.",
            "Once ROI locks, the pipeline switches to:",
            "ROI crop -> YOLO -> tracking -> cropper -> selection -> VLM",
            f"locked={roi_pkg['roi_layer_locked']}",
            f"bounds={roi_pkg['roi_layer_bounds']}",
        ],
        (18, 72),
        scale=0.38,
        line_gap=22,
    )
    return cv2.hconcat([left, panel])


def main() -> None:
    defaults = load_runtime_settings()
    parser = argparse.ArgumentParser(description="Upson ROI + VLM visualizer with clip window and realtime pacing")
    parser.add_argument("--input-source", default=defaults["input_source"], choices=["video", "camera"])
    parser.add_argument("--video", default=defaults["video"])
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--conf", type=float, default=defaults["conf"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--vlm-device", default="", help="Override device for VLM only")
    parser.add_argument("--roi-enabled", action="store_true", default=defaults["roi_enabled"])
    parser.add_argument("--roi-threshold", type=int, default=defaults["roi_threshold"])
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--gstreamer", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", default=defaults["output"], help="Optional output video path")
    parser.add_argument("--start-seconds", type=float, default=DEFAULT_START_SECONDS)
    parser.add_argument("--end-seconds", type=float, default=DEFAULT_END_SECONDS)
    parser.add_argument("--max-queue-size", type=int, default=64)
    parser.add_argument(
        "--no-realtime-throttle",
        action="store_true",
        default=False,
        help="Run as fast as possible instead of pacing the display/output to the source FPS.",
    )
    args = parser.parse_args()

    frame_resolution = tuple(defaults["frame_resolution"])
    fps, width, height = probe_video_metadata(args.input_source, args.video, frame_resolution)
    start_frame = max(0, int(round(float(args.start_seconds) * fps)))
    end_frame = max(start_frame, int(round(float(args.end_seconds) * fps)))

    output_w = int(width * DISPLAY_SCALE) + 780
    output_h = int(height * DISPLAY_SCALE)
    out = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps or 30.0, (output_w, output_h))
        if not out.isOpened():
            raise RuntimeError(f"Could not create output video: {args.output}")

    vlm_device = args.vlm_device.strip() or args.device
    vlm_state = initialize_vlm_layer(
        VLMConfig(
            config_vlm_enabled=bool(defaults["vlm_enabled"]),
            config_vlm_model=defaults["vlm_model"],
            config_device=vlm_device,
        )
    )

    input_layer = InputLayer()
    input_layer.initialize_input_layer(
        config_input_source=args.input_source,
        config_frame_resolution=frame_resolution,
        config_input_path=args.video,
        camera_device_index=args.camera_index,
        use_gstreamer=args.gstreamer,
    )
    initialize_yolo_layer(model_name=args.model, conf_threshold=args.conf, device=args.device)
    initialize_roi_layer(config_roi_enabled=args.roi_enabled, config_roi_vehicle_count_threshold=args.roi_threshold)
    initialize_tracking_layer(frame_rate=int(fps))
    crop_cache_state = initialize_vlm_crop_cache(defaults["vlm_crop_cache_size"], defaults["vlm_dead_after_lost_frames"])
    initialize_vehicle_state_layer(prune_after_lost_frames=None)
    worker = AsyncVLMWorker(
        vlm_state=vlm_state,
        feedback_enabled=bool(defaults["vlm_crop_feedback_enabled"]),
        max_queue_size=args.max_queue_size,
    )
    worker.start()
    debug_state: dict[str, dict[str, Any]] = {}

    clip_started = False
    realtime_start_wall = 0.0
    clip_frame_index = 0

    try:
        while True:
            for result in worker.drain_results():
                ack = result["ack"]
                register_vlm_ack_package(crop_cache_state, ack)
                update_vehicle_state_from_vlm_ack(ack)
                if ack.vlm_ack_status == "accepted" and result["raw_result"] is not None:
                    update_vehicle_state_from_vlm(build_vlm_layer_package(result["raw_result"]))
                normalized = result["normalized_result"] or {}
                debug_state[result["track_id"]] = {
                    "prompt": result["prompt"],
                    "raw_response": result["raw_result"].vlm_layer_raw_text.strip() if result["raw_result"] else f"ERROR: {result['error_text']}",
                    "normalized_result": result["normalized_result"],
                    "ack_status": ack.vlm_ack_status,
                    "ack_reason": ack.vlm_ack_reason,
                    "retry_reasons": normalized.get("vlm_retry_reasons", []) if defaults["vlm_crop_feedback_enabled"] else [],
                    "progressed_only_tracking": ack.vlm_ack_status == "accepted",
                    "runtime_sec": result["runtime_sec"],
                    "dispatch_frame_id": result["dispatch_frame_id"],
                    "query_type": result["query_type"],
                }

            raw_frame = input_layer.read_next_frame()
            if raw_frame is None:
                break

            input_package = input_layer.build_input_layer_package(raw_frame)
            source_frame_id = int(input_package.input_layer_frame_id)
            if source_frame_id < start_frame:
                continue
            if source_frame_id > end_frame:
                break

            if not clip_started:
                clip_started = True
                realtime_start_wall = time.time()

            clip_frame_index += 1
            input_pkg = input_package_to_dict(input_package)

            calibration_raw = run_yolo_detection(input_pkg)
            calibration_dets = filter_yolo_detections(calibration_raw)
            calibration_yolo_pkg = build_yolo_layer_package(source_frame_id, calibration_dets)
            roi_state = update_roi_state(input_package, calibration_yolo_pkg["yolo_layer_detections"])
            roi_frame = apply_roi_to_frame(input_package)
            roi_pkg = build_roi_layer_package(input_package, roi_frame)

            if not roi_pkg["roi_layer_locked"]:
                fps_text = fps if not args.no_realtime_throttle and fps > 0 else (clip_frame_index / max(1e-9, (time.time() - realtime_start_wall)))
                canvas = build_calibration_canvas(
                    frame_view=input_package.input_layer_image.copy(),
                    calibration_detections=calibration_yolo_pkg["yolo_layer_detections"],
                    roi_state=roi_state,
                    roi_pkg=roi_pkg,
                    fps_text=fps_text,
                    frame_id=source_frame_id,
                    clip_frame_index=clip_frame_index,
                    target_count=args.roi_threshold,
                )
            else:
                roi_raw = run_yolo_detection(roi_pkg)
                roi_filtered = filter_yolo_detections(roi_raw)
                roi_yolo_pkg = build_yolo_layer_package(roi_pkg["roi_layer_frame_id"], roi_filtered)
                current_tracks = update_tracks(roi_yolo_pkg)
                status_tracks = assign_tracking_status(current_tracks, roi_pkg["roi_layer_frame_id"])
                tracking_pkg = build_tracking_layer_package(roi_pkg["roi_layer_frame_id"], status_tracks)
                update_vehicle_state_from_tracking(tracking_pkg)

                frame_view = input_package.input_layer_image.copy()
                if roi_pkg["roi_layer_bounds"] is not None:
                    x1, y1, x2, y2 = [int(v) for v in roi_pkg["roi_layer_bounds"]]
                    cv2.rectangle(frame_view, (x1, y1), (x2, y2), LOCKED_COLOR, 3)
                dispatch_track_ids: list[str] = []
                roi_input_pkg = {
                    "input_layer_frame_id": roi_pkg["roi_layer_frame_id"],
                    "input_layer_timestamp": roi_pkg["roi_layer_timestamp"],
                    "input_layer_image": roi_pkg["roi_layer_image"],
                    "input_layer_source_type": "roi_crop",
                    "input_layer_resolution": roi_pkg["roi_layer_image"].shape[1::-1],
                }

                for index, row in enumerate(tracking_rows(tracking_pkg)):
                    refresh_vlm_crop_cache_track_state(crop_cache_state, row, roi_pkg["roi_layer_frame_id"])
                    vehicle_record_for_row = get_vehicle_state_record(row["track_id"])
                    if vehicle_record_for_row and vehicle_record_for_row.get("vehicle_state_layer_vlm_called"):
                        debug_state.setdefault(str(row["track_id"]), {})["progressed_only_tracking"] = True
                        continue
                    if row["status"] == "lost":
                        continue
                    request_pkg = build_vlm_frame_cropper_request_package(
                        input_layer_package=roi_input_pkg,
                        tracking_layer_package=tracking_pkg,
                        track_index=index,
                        vlm_frame_cropper_trigger_reason=f'tracking_status:{row["status"]}',
                        config_vlm_enabled=True,
                    )
                    if request_pkg is None:
                        continue
                    crop = extract_vlm_object_crop(roi_input_pkg, request_pkg)
                    crop_pkg = build_vlm_frame_cropper_package(request_pkg, crop)
                    update_vlm_crop_cache(crop_cache_state, row, crop_pkg, roi_pkg["roi_layer_frame_id"], f'tracking_status:{row["status"]}')

                if vlm_state.config_vlm_enabled:
                    for tid in sorted(crop_cache_state["track_caches"].keys(), key=lambda x: (len(str(x)), str(x))):
                        vehicle_record_for_dispatch = get_vehicle_state_record(tid)
                        if vehicle_record_for_dispatch and vehicle_record_for_dispatch.get("vehicle_state_layer_vlm_called"):
                            continue
                        dispatch = build_vlm_dispatch_package(crop_cache_state, tid)
                        if dispatch is None:
                            continue
                        dispatch_track_ids.append(str(tid))
                        inner = dispatch["vlm_frame_cropper_layer_package"]
                        vlm_crop_pkg = VLMFrameCropperLayerPackage(
                            vlm_frame_cropper_layer_track_id=str(inner["vlm_frame_cropper_layer_track_id"]),
                            vlm_frame_cropper_layer_image=inner["vlm_frame_cropper_layer_image"],
                            vlm_frame_cropper_layer_bbox=tuple(int(x) for x in inner["vlm_frame_cropper_layer_bbox"]),
                        )
                        query_type = DEFAULT_VLM_QUERY_TYPE if defaults["vlm_crop_feedback_enabled"] else "vehicle_semantics_single_shot_v1"
                        if dispatch["vlm_dispatch_mode"] == "dead_best_available":
                            query_type = "vehicle_semantics_single_shot_v1"
                        worker.submit(
                            {
                                "track_id": str(tid),
                                "dispatch_frame_id": int(roi_pkg["roi_layer_frame_id"]),
                                "prompt_text": prepare_vlm_prompt(query_type, vlm_crop_pkg),
                                "query_type": query_type,
                                "submitted_at": time.time(),
                                "vlm_crop_pkg": vlm_crop_pkg,
                            }
                        )

                focus_track_id = pick_focus_track(crop_cache_state, dispatch_track_ids)
                vehicle_record = get_vehicle_state_record(focus_track_id) if focus_track_id is not None else None
                fps_text = fps if not args.no_realtime_throttle and fps > 0 else (clip_frame_index / max(1e-9, (time.time() - realtime_start_wall)))
                display_tracking_pkg = translate_tracking_pkg_to_full_frame(tracking_pkg, roi_pkg["roi_layer_bounds"])
                canvas = build_canvas(
                    frame_view=frame_view,
                    tracking_pkg=display_tracking_pkg,
                    focus_track_id=focus_track_id,
                    crop_cache_state=crop_cache_state,
                    debug_state=debug_state,
                    vehicle_record=vehicle_record,
                    cache_size=defaults["vlm_crop_cache_size"],
                    fps_text=fps_text,
                    frame_id=source_frame_id,
                    feedback_enabled=bool(defaults["vlm_crop_feedback_enabled"]),
                )
                draw_text(canvas, "ROI LOCKED PIPELINE", 16, 28, color=LOCKED_COLOR, scale=0.54)
                draw_text(canvas, f"bounds={roi_pkg['roi_layer_bounds']}", 16, 50, color=LOCKED_COLOR, scale=0.28)
                draw_text(canvas, f"clip=0:48-1:48  src_frame={source_frame_id}", 16, 72, color=LOCKED_COLOR, scale=0.28)
                overlay_async_status(canvas, worker.get_status(), roi_pkg["roi_layer_frame_id"])

            if out is not None:
                out.write(canvas)
            if args.show:
                cv2.imshow("ROI VLM Upson visualizer", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames > 0 and clip_frame_index >= args.max_frames:
                break
            if not args.no_realtime_throttle:
                maybe_sleep_to_source_fps(realtime_start_wall, clip_frame_index, fps)
    finally:
        worker.shutdown()
        input_layer.close_input_layer()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

    if args.output:
        print(f"Saved ROI VLM Upson visualization to: {args.output}")


if __name__ == "__main__":
    main()
