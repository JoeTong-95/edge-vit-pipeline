#!/usr/bin/env python3
"""
visualize_vlm_roi.py
Show ROI calibration first, then run the tracking -> cropper -> selection -> VLM
sequence only inside the locked ROI crop.
"""

from __future__ import annotations

import argparse
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
_ROI_DIR = _THIS_DIR.parent / "roi-layer"

for path in (_THIS_DIR, _CONFIG_DIR, _INPUT_DIR, _YOLO_DIR, _TRACKING_DIR, _CROPPER_LAYER_DIR, _VEHICLE_STATE_DIR, _ROI_DIR):
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
DETECTION_COLOR = (0, 255, 0)
PANEL_BG = (20, 20, 24)


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
        "vlm_dead_after_lost_frames": get_config_value(config, "config_vlm_dead_after_lost_frames"),
        "roi_enabled": get_config_value(config, "config_roi_enabled"),
        "roi_threshold": get_config_value(config, "config_roi_vehicle_count_threshold"),
        "output": "",
    }


def draw_detection_boxes(frame, detections):
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection["yolo_detection_bbox"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), DETECTION_COLOR, 2)


def draw_roi_bounds(frame, roi_bounds, locked):
    if roi_bounds is None:
        return
    x1, y1, x2, y2 = [int(value) for value in roi_bounds]
    color = LOCKED_COLOR if locked else ROI_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if locked else 2)
    draw_text_block(frame, ["ROI LOCKED" if locked else "ROI CALIBRATING"], (x1 + 6, max(22, y1 - 10)), color=color, scale=0.30, line_gap=16)


def build_calibration_canvas(
    frame_view,
    calibration_detections,
    roi_state,
    roi_pkg,
    fps_text,
    frame_id,
    target_count,
):
    left = frame_view.copy()
    draw_detection_boxes(left, calibration_detections)
    draw_roi_bounds(left, roi_state["roi_layer_bounds"], roi_state["roi_layer_locked"])
    overlay = left.copy()
    cv2.rectangle(overlay, (10, 10), (350, 132), PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.65, left, 0.35, 0, left)
    draw_text_block(
        left,
        [
            f"Frame: {frame_id}",
            "Phase: ROI calibration",
            f"Threshold: {target_count}",
            f"Candidate count: {roi_state['roi_candidate_box_count']}",
            f"Current detections: {len(calibration_detections)}",
            f"FPS: {fps_text:.1f}",
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
            "Once ROI locks, this visualizer switches to:",
            "ROI crop -> YOLO -> tracking -> cropper -> selection -> VLM",
            f"locked={roi_pkg['roi_layer_locked']}",
            f"bounds={roi_pkg['roi_layer_bounds']}",
        ],
        (18, 72),
        scale=0.38,
        line_gap=22,
    )
    return np.hstack([left, panel])


def main() -> None:
    defaults = load_runtime_settings()
    parser = argparse.ArgumentParser(description="Visualize ROI calibration, then ROI-local cropper selection and VLM")
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
    args = parser.parse_args()

    frame_resolution = tuple(defaults["frame_resolution"])
    fps, width, height = probe_video_metadata(args.input_source, args.video, frame_resolution)
    output_w = int(width * DISPLAY_SCALE) + 780
    output_h = int(height * DISPLAY_SCALE)
    out = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_w, output_h))
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
    debug_state: dict[str, dict[str, Any]] = {}
    start_time = time.time()

    try:
        while True:
            raw_frame = input_layer.read_next_frame()
            if raw_frame is None:
                break

            input_package = input_layer.build_input_layer_package(raw_frame)
            input_pkg = input_package_to_dict(input_package)

            calibration_raw = run_yolo_detection(input_pkg)
            calibration_dets = filter_yolo_detections(calibration_raw)
            calibration_yolo_pkg = build_yolo_layer_package(input_package.input_layer_frame_id, calibration_dets)
            roi_state = update_roi_state(input_package, calibration_yolo_pkg["yolo_layer_detections"])
            roi_frame = apply_roi_to_frame(input_package)
            roi_pkg = build_roi_layer_package(input_package, roi_frame)

            elapsed = time.time() - start_time
            fps_text = input_package.input_layer_frame_id / elapsed if elapsed > 0 else 0.0

            if not roi_pkg["roi_layer_locked"]:
                canvas = build_calibration_canvas(
                    frame_view=input_package.input_layer_image.copy(),
                    calibration_detections=calibration_yolo_pkg["yolo_layer_detections"],
                    roi_state=roi_state,
                    roi_pkg=roi_pkg,
                    fps_text=fps_text,
                    frame_id=input_package.input_layer_frame_id,
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

                frame_view = roi_pkg["roi_layer_image"].copy()
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
                        try:
                            query_type = DEFAULT_VLM_QUERY_TYPE if defaults["vlm_crop_feedback_enabled"] else "vehicle_semantics_single_shot_v1"
                            if dispatch["vlm_dispatch_mode"] == "dead_best_available":
                                query_type = "vehicle_semantics_single_shot_v1"
                            raw_result = run_vlm_inference(vlm_state, vlm_crop_pkg, query_type)
                            normalized = normalize_vlm_result(raw_result)
                            ack = build_vlm_ack_package_from_result(raw_result) if defaults["vlm_crop_feedback_enabled"] or query_type == "vehicle_semantics_single_shot_v1" else build_vlm_ack_package(str(tid), "accepted", "feedback_disabled_single_shot", False)
                            register_vlm_ack_package(crop_cache_state, ack)
                            update_vehicle_state_from_vlm_ack(ack)
                            if ack.vlm_ack_status == "accepted":
                                update_vehicle_state_from_vlm(build_vlm_layer_package(raw_result))
                            debug_state[str(tid)] = {
                                "prompt": prepare_vlm_prompt(query_type, vlm_crop_pkg),
                                "raw_response": raw_result.vlm_layer_raw_text.strip(),
                                "normalized_result": normalized,
                                "ack_status": ack.vlm_ack_status,
                                "ack_reason": ack.vlm_ack_reason,
                                "retry_reasons": normalized.get("vlm_retry_reasons", []) if defaults["vlm_crop_feedback_enabled"] else [],
                                "progressed_only_tracking": ack.vlm_ack_status == "accepted",
                            }
                        except Exception as exc:
                            ack = build_vlm_ack_package(str(tid), "retry_requested", f"VLM error: {exc}", True)
                            register_vlm_ack_package(crop_cache_state, ack)
                            update_vehicle_state_from_vlm_ack(ack)
                            debug_state[str(tid)] = {
                                "prompt": "",
                                "raw_response": f"ERROR: {exc}",
                                "normalized_result": None,
                                "ack_status": "retry_requested",
                                "ack_reason": f"VLM error: {exc}",
                                "retry_reasons": ["occluded"],
                            }

                focus_track_id = pick_focus_track(crop_cache_state, dispatch_track_ids)
                vehicle_record = get_vehicle_state_record(focus_track_id) if focus_track_id is not None else None
                canvas = build_canvas(
                    frame_view=frame_view,
                    tracking_pkg=tracking_pkg,
                    focus_track_id=focus_track_id,
                    crop_cache_state=crop_cache_state,
                    debug_state=debug_state,
                    vehicle_record=vehicle_record,
                    cache_size=defaults["vlm_crop_cache_size"],
                    fps_text=fps_text,
                    frame_id=roi_pkg["roi_layer_frame_id"],
                    feedback_enabled=bool(defaults["vlm_crop_feedback_enabled"]),
                )
                draw_text(canvas, "ROI LOCKED PIPELINE", 16, 28, color=LOCKED_COLOR, scale=0.54)
                draw_text(canvas, f"bounds={roi_pkg['roi_layer_bounds']}", 16, 50, color=LOCKED_COLOR, scale=0.28)

            if out is not None:
                out.write(canvas)
            if args.show:
                cv2.imshow("VLM ROI visualizer", canvas)
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
        print(f"Saved VLM ROI visualization to: {args.output}")


if __name__ == "__main__":
    main()
