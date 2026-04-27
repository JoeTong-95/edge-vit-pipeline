#!/usr/bin/env python3
"""
visualize_roi_vlm.py
Show ROI calibration first, then run YOLO + VLM only inside the locked ROI.

Run from project root:
    python src/roi-layer/util/visualize_roi_vlm.py --show
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
_CONFIG_DIR = _SRC_DIR / "configuration-layer"
_INPUT_DIR = _SRC_DIR / "input-layer"
_YOLO_DIR = _SRC_DIR / "yolo-layer"
_VLM_DIR = _SRC_DIR / "vlm-layer"
_CROPPER_DIR = _SRC_DIR / "vlm-frame-cropper-layer"
_REPO_ROOT = _SRC_DIR.parent
for path in (_THIS_DIR, _ROI_DIR, _CONFIG_DIR, _INPUT_DIR, _YOLO_DIR, _VLM_DIR, _CROPPER_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import cv2
import numpy as np
from config_node import get_config_value, load_config, validate_config
from detector import build_yolo_layer_package, filter_yolo_detections, initialize_yolo_layer, run_yolo_detection
from input_layer import InputLayer
from layer import (
    VLMConfig,
    VLMFrameCropperLayerPackage,
    initialize_vlm_layer,
    normalize_vlm_result,
    prepare_vlm_prompt,
    run_vlm_inference,
)
from roi_layer import apply_roi_to_frame, build_roi_layer_package, initialize_roi_layer, update_roi_state
from vlm_frame_cropper_layer import (
    build_vlm_frame_cropper_package,
    build_vlm_frame_cropper_request_package,
    extract_vlm_object_crop,
)

FRAME_TEXT = (245, 245, 245)
SHADOW_TEXT = (0, 0, 0)
ROI_COLOR = (0, 200, 255)
LOCKED_COLOR = (0, 220, 120)
DETECTION_COLOR = (0, 255, 0)
SELECTED_COLOR = (255, 255, 255)
PANEL_BG = (20, 20, 24)
CARD_BG = (34, 34, 40)
MUTED = (180, 180, 185)
GOOD = (70, 210, 110)
BAD = (90, 90, 255)
TEXT_SUPERSAMPLE = 2


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
        "roi_enabled": get_config_value(config, "config_roi_enabled"),
        "roi_threshold": get_config_value(config, "config_roi_vehicle_count_threshold"),
        "vlm_enabled": get_config_value(config, "config_vlm_enabled"),
        "vlm_backend": get_config_value(config, "config_vlm_backend"),
        "vlm_model": _resolve_vlm_model_path(get_config_value(config, "config_vlm_model")),
        "output": "",
    }


def probe_video_metadata(input_source: str, input_path: str, frame_resolution: tuple[int, int]) -> tuple[float, int, int]:
    width, height = frame_resolution
    fps = 30.0
    if input_source == "video":
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not probe video source: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    return fps, width, height


def maybe_sleep_to_source_fps(start_wall: float, frame_id: int, fps: float) -> None:
    if fps <= 0:
        return
    target_elapsed = frame_id / fps
    sleep_sec = target_elapsed - (time.time() - start_wall)
    if sleep_sec > 0:
        time.sleep(sleep_sec)


def input_package_to_dict(package: Any) -> dict[str, Any]:
    return {
        "input_layer_frame_id": package.input_layer_frame_id,
        "input_layer_timestamp": package.input_layer_timestamp,
        "input_layer_image": package.input_layer_image,
        "input_layer_source_type": package.input_layer_source_type,
        "input_layer_resolution": package.input_layer_resolution,
    }


def draw_text_block(frame: np.ndarray, lines: list[str], origin: tuple[int, int], color: tuple[int, int, int] = FRAME_TEXT, scale: float = 0.42, line_gap: int = 22) -> None:
    if not lines:
        return
    h, w = frame.shape[:2]
    overlay = np.zeros((h * TEXT_SUPERSAMPLE, w * TEXT_SUPERSAMPLE, 3), dtype=np.uint8)
    x, y = origin
    x *= TEXT_SUPERSAMPLE
    y *= TEXT_SUPERSAMPLE
    super_scale = scale * TEXT_SUPERSAMPLE
    super_gap = line_gap * TEXT_SUPERSAMPLE
    shadow_offset = 2 * TEXT_SUPERSAMPLE
    for line in lines:
        cv2.putText(overlay, line, (x + shadow_offset, y + shadow_offset), cv2.FONT_HERSHEY_SIMPLEX, super_scale, SHADOW_TEXT, 4, cv2.LINE_AA)
        cv2.putText(overlay, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, super_scale, color, 2, cv2.LINE_AA)
        y += super_gap
    resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
    mask = np.any(resized > 0, axis=2)
    frame[mask] = resized[mask]


def draw_boxes(frame: np.ndarray, detections: list[dict[str, Any]], selected_index: int | None = None) -> None:
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = [int(value) for value in detection["yolo_detection_bbox"]]
        color = SELECTED_COLOR if selected_index == index else DETECTION_COLOR
        thickness = 3 if selected_index == index else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f'{detection["yolo_detection_class"]} {detection["yolo_detection_confidence"]:.2f}'
        draw_text_block(frame, [label], (x1 + 4, max(18, y1 - 8)), color=color, scale=0.26, line_gap=14)


def draw_roi_bounds(frame: np.ndarray, roi_bounds: tuple[int, int, int, int] | None, locked: bool) -> None:
    if roi_bounds is None:
        return
    x1, y1, x2, y2 = [int(value) for value in roi_bounds]
    color = LOCKED_COLOR if locked else ROI_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if locked else 2)
    draw_text_block(frame, ["ROI LOCKED" if locked else "ROI CALIBRATING"], (x1 + 6, max(20, y1 - 10)), color=color, scale=0.30, line_gap=16)


def paste_with_aspect(canvas: np.ndarray, image: np.ndarray | None, x: int, y: int, w: int, h: int) -> None:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), CARD_BG, -1)
    if image is None or getattr(image, "size", 0) == 0:
        draw_text_block(canvas, ["none"], (x + 10, y + h // 2), color=MUTED, scale=0.34, line_gap=16)
        return
    ih, iw = image.shape[:2]
    scale = min(max(1, w - 8) / iw, max(1, h - 8) / ih)
    dw = max(1, int(iw * scale))
    dh = max(1, int(ih * scale))
    resized = cv2.resize(image, (dw, dh), interpolation=cv2.INTER_LINEAR)
    ox = x + (w - dw) // 2
    oy = y + (h - dh) // 2
    canvas[oy:oy + dh, ox:ox + dw] = resized
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (110, 110, 120), 1)


def select_best_detection(detections: list[dict[str, Any]]) -> int | None:
    if not detections:
        return None
    best_index = None
    best_score = None
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["yolo_detection_bbox"]
        area = max(1.0, (x2 - x1) * (y2 - y1))
        score = float(detection["yolo_detection_confidence"]) + (0.35 * min(1.0, area / 50000.0))
        if best_score is None or score > best_score:
            best_index = index
            best_score = score
    return best_index


def build_roi_input_package(roi_pkg: dict[str, Any]) -> dict[str, Any]:
    roi_image = roi_pkg["roi_layer_image"]
    return {
        "input_layer_frame_id": roi_pkg["roi_layer_frame_id"],
        "input_layer_timestamp": roi_pkg["roi_layer_timestamp"],
        "input_layer_image": roi_image,
        "input_layer_source_type": "roi_crop",
        "input_layer_resolution": (roi_image.shape[1], roi_image.shape[0]),
    }


def build_single_detection_tracking_package(roi_pkg: dict[str, Any], detection: dict[str, Any]) -> dict[str, Any]:
    return {
        "tracking_layer_frame_id": roi_pkg["roi_layer_frame_id"],
        "tracking_layer_track_id": ["roi-focus"],
        "tracking_layer_bbox": [list(detection["yolo_detection_bbox"])],
        "tracking_layer_detector_class": [detection["yolo_detection_class"]],
        "tracking_layer_confidence": [float(detection["yolo_detection_confidence"])],
        "tracking_layer_status": ["active"],
    }


def make_canvas(
    annotated_full_frame: np.ndarray,
    roi_view: np.ndarray,
    roi_pkg: dict[str, Any],
    roi_state: dict[str, Any],
    calibration_detection_count: int,
    roi_detections: list[dict[str, Any]],
    selected_index: int | None,
    selected_crop: np.ndarray | None,
    latest_vlm: dict[str, Any] | None,
    target_count: int,
    fps_text: float,
) -> np.ndarray:
    full_h, full_w = annotated_full_frame.shape[:2]
    roi_panel_w = max(360, full_w // 2)
    info_panel_w = 430
    canvas = np.zeros((full_h, full_w + roi_panel_w + info_panel_w, 3), dtype=np.uint8)
    canvas[:, :full_w] = annotated_full_frame

    roi_panel = np.zeros((full_h, roi_panel_w, 3), dtype=np.uint8)
    roi_panel[:] = PANEL_BG
    phase = "ROI CALIBRATION" if not roi_pkg["roi_layer_locked"] else "ROI LOCKED -> YOLO + VLM"
    phase_color = ROI_COLOR if not roi_pkg["roi_layer_locked"] else LOCKED_COLOR
    draw_text_block(roi_panel, [phase], (14, 28), color=phase_color, scale=0.58, line_gap=24)
    if roi_view.size > 0:
        preview_h = min(full_h - 110, roi_view.shape[0])
        preview_w = roi_panel_w - 28
        scale = min(preview_w / roi_view.shape[1], preview_h / roi_view.shape[0])
        disp_w = max(1, int(roi_view.shape[1] * scale))
        disp_h = max(1, int(roi_view.shape[0] * scale))
        roi_preview = cv2.resize(roi_view, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        x_offset = 14 + (preview_w - disp_w) // 2
        y_offset = 56
        roi_panel[y_offset:y_offset + disp_h, x_offset:x_offset + disp_w] = roi_preview
        cv2.rectangle(roi_panel, (x_offset, y_offset), (x_offset + disp_w, y_offset + disp_h), phase_color, 2)
    draw_text_block(
        roi_panel,
        [
            f"locked={roi_pkg['roi_layer_locked']}",
            f"bounds={roi_pkg['roi_layer_bounds']}",
            f"calibration detections={calibration_detection_count}",
            f"roi detections={len(roi_detections)}",
            f"fps={fps_text:.1f}",
        ],
        (14, full_h - 110),
        scale=0.34,
        line_gap=18,
    )

    info_panel = np.zeros((full_h, info_panel_w, 3), dtype=np.uint8)
    info_panel[:] = PANEL_BG
    draw_text_block(info_panel, ["ROI -> VLM VIEW"], (14, 28), scale=0.58, line_gap=24)
    draw_text_block(
        info_panel,
        [
            f"threshold={target_count}",
            f"candidate count={roi_state['roi_candidate_box_count']}",
            "full-frame YOLO is only used for ROI calibration.",
            "After ROI lock, YOLO runs on ROI crop only.",
        ],
        (14, 64),
        scale=0.34,
        line_gap=18,
    )

    draw_text_block(info_panel, ["Selected ROI crop"], (14, 156), scale=0.40, line_gap=18)
    paste_with_aspect(info_panel, selected_crop, 14, 174, info_panel_w - 28, 180)

    vlm_lines = ["No VLM result yet."]
    vlm_color = MUTED
    if latest_vlm is not None:
        if latest_vlm.get("error"):
            vlm_lines = [f"error={latest_vlm['error']}"]
            vlm_color = BAD
        else:
            normalized = latest_vlm["normalized"]
            vlm_lines = [
                f"frame={latest_vlm['frame_id']}",
                f"label={latest_vlm['label']}",
                f"is_target_vehicle={normalized.get('is_target_vehicle')}",
                f"truck_type={normalized.get('truck_type')}",
                f"confidence={normalized.get('vlm_layer_confidence')}",
                f"raw={latest_vlm['raw_text'][:120]}",
            ]
            vlm_color = GOOD if normalized.get("is_target_vehicle") is not False else BAD
    draw_text_block(info_panel, ["Latest VLM result"], (14, 382), scale=0.40, line_gap=18)
    draw_text_block(info_panel, vlm_lines, (14, 408), color=vlm_color, scale=0.33, line_gap=18)

    canvas[:, full_w:full_w + roi_panel_w] = roi_panel
    canvas[:, full_w + roi_panel_w:] = info_panel
    return canvas


def main() -> None:
    defaults = load_runtime_settings()
    parser = argparse.ArgumentParser(description="Visualize ROI calibration, then ROI-only YOLO + VLM")
    parser.add_argument("--input-source", default=defaults["input_source"], choices=["video", "camera"])
    parser.add_argument("--video", default=defaults["video"])
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--conf", type=float, default=defaults["conf"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--vlm-device", default="", help="Override device for VLM only")
    parser.add_argument("--roi-enabled", action="store_true", default=defaults["roi_enabled"])
    parser.add_argument("--roi-threshold", type=int, default=defaults["roi_threshold"])
    parser.add_argument("--vlm-every-n-frames", type=int, default=15, help="Run VLM every N locked ROI frames")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--gstreamer", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", default=defaults["output"], help="Optional output video path")
    parser.add_argument(
        "--no-realtime-throttle",
        action="store_true",
        default=False,
        help="Run as fast as possible instead of pacing the display/output to the source FPS.",
    )
    args = parser.parse_args()

    frame_resolution = tuple(defaults["frame_resolution"])
    fps, width, height = probe_video_metadata(args.input_source, args.video, frame_resolution)
    roi_panel_w = max(360, width // 2)
    info_panel_w = 430
    out = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width + roi_panel_w + info_panel_w, height))
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
    initialize_roi_layer(config_roi_enabled=args.roi_enabled, config_roi_vehicle_count_threshold=args.roi_threshold)

    vlm_state = initialize_vlm_layer(
        VLMConfig(
            config_vlm_enabled=bool(defaults["vlm_enabled"]),
            config_vlm_backend=defaults["vlm_backend"],
            config_vlm_model=defaults["vlm_model"],
            config_device=args.vlm_device.strip() or args.device,
        )
    )

    last_vlm_frame_id = -10**9
    latest_vlm: dict[str, Any] | None = None
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

            annotated = input_package.input_layer_image.copy()
            if not roi_pkg["roi_layer_locked"]:
                draw_boxes(annotated, calibration_yolo_pkg["yolo_layer_detections"])
            draw_roi_bounds(annotated, roi_state["roi_layer_bounds"], roi_state["roi_layer_locked"])

            roi_detections: list[dict[str, Any]] = []
            selected_index = None
            selected_crop = None
            roi_display = roi_pkg["roi_layer_image"].copy()
            if roi_pkg["roi_layer_locked"]:
                roi_raw = run_yolo_detection(roi_pkg)
                roi_detections = filter_yolo_detections(roi_raw)
                selected_index = select_best_detection(roi_detections)
                draw_boxes(roi_display, roi_detections, selected_index=selected_index)

                if selected_index is not None:
                    detection = roi_detections[selected_index]
                    roi_input_pkg = build_roi_input_package(roi_pkg)
                    pseudo_tracking_pkg = build_single_detection_tracking_package(roi_pkg, detection)
                    request_pkg = build_vlm_frame_cropper_request_package(
                        input_layer_package=roi_input_pkg,
                        tracking_layer_package=pseudo_tracking_pkg,
                        track_index=0,
                        vlm_frame_cropper_trigger_reason="roi_locked_detection",
                        config_vlm_enabled=True,
                    )
                    selected_crop = extract_vlm_object_crop(roi_input_pkg, request_pkg)

                    if (
                        defaults["vlm_enabled"]
                        and (input_package.input_layer_frame_id - last_vlm_frame_id) >= max(1, int(args.vlm_every_n_frames))
                    ):
                        crop_pkg = build_vlm_frame_cropper_package(request_pkg, selected_crop)
                        vlm_crop_pkg = VLMFrameCropperLayerPackage(
                            vlm_frame_cropper_layer_track_id=str(crop_pkg["vlm_frame_cropper_layer_track_id"]),
                            vlm_frame_cropper_layer_image=crop_pkg["vlm_frame_cropper_layer_image"],
                            vlm_frame_cropper_layer_bbox=tuple(int(v) for v in crop_pkg["vlm_frame_cropper_layer_bbox"]),
                        )
                        try:
                            raw_result = run_vlm_inference(vlm_state, vlm_crop_pkg, "vehicle_semantics_single_shot_v1")
                            normalized = normalize_vlm_result(raw_result)
                            latest_vlm = {
                                "frame_id": input_package.input_layer_frame_id,
                                "prompt": prepare_vlm_prompt("vehicle_semantics_single_shot_v1", vlm_crop_pkg),
                                "raw_text": raw_result.vlm_layer_raw_text.strip(),
                                "normalized": normalized,
                                "label": detection["yolo_detection_class"],
                                "error": "",
                            }
                        except Exception as exc:
                            latest_vlm = {
                                "frame_id": input_package.input_layer_frame_id,
                                "prompt": "",
                                "raw_text": "",
                                "normalized": {},
                                "label": detection["yolo_detection_class"],
                                "error": str(exc),
                            }
                        last_vlm_frame_id = input_package.input_layer_frame_id

            elapsed = time.time() - start_time
            processing_fps = input_package.input_layer_frame_id / elapsed if elapsed > 0 else 0.0
            fps_text = fps if not args.no_realtime_throttle and fps > 0 else processing_fps
            draw_text_block(
                annotated,
                [
                    f"Frame: {input_package.input_layer_frame_id}",
                    f"ROI locked: {roi_pkg['roi_layer_locked']}",
                    f"Threshold: {args.roi_threshold}",
                    f"VLM every N: {max(1, int(args.vlm_every_n_frames))}",
                    f"Feed FPS: {fps_text:.1f}",
                ],
                (12, 24),
                scale=0.40,
                line_gap=22,
            )

            canvas = make_canvas(
                annotated_full_frame=annotated,
                roi_view=roi_display,
                roi_pkg=roi_pkg,
                roi_state=roi_state,
                calibration_detection_count=len(calibration_yolo_pkg["yolo_layer_detections"]),
                roi_detections=roi_detections,
                selected_index=selected_index,
                selected_crop=selected_crop,
                latest_vlm=latest_vlm,
                target_count=args.roi_threshold,
                fps_text=fps_text,
            )

            if out is not None:
                out.write(canvas)
            if args.show:
                cv2.imshow("ROI -> VLM Visualizer", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames > 0 and input_package.input_layer_frame_id >= args.max_frames:
                break
            if not args.no_realtime_throttle:
                maybe_sleep_to_source_fps(start_time, input_package.input_layer_frame_id, fps)
    finally:
        input_layer.close_input_layer()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

    if args.output:
        print(f"Saved ROI -> VLM visualization to: {args.output}")


if __name__ == "__main__":
    main()
