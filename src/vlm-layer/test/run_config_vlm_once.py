from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_VLM_DIR = _THIS_DIR.parent
_SRC_DIR = _VLM_DIR.parent
_REPO_ROOT = _SRC_DIR.parent
_CONFIG_DIR = _SRC_DIR / "configuration-layer"
_INPUT_DIR = _SRC_DIR / "input-layer"
_YOLO_DIR = _SRC_DIR / "yolo-layer"
_TRACKING_DIR = _SRC_DIR / "tracking-layer"
_CROPPER_DIR = _SRC_DIR / "vlm-frame-cropper-layer"

for path in (_THIS_DIR, _VLM_DIR, _CONFIG_DIR, _INPUT_DIR, _YOLO_DIR, _TRACKING_DIR, _CROPPER_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from config_node import get_config_value, load_config, validate_config
from detector import build_yolo_layer_package, filter_yolo_detections, initialize_yolo_layer, run_yolo_detection
from input_layer import InputLayer
from layer import (
    DEFAULT_VLM_DEBUG_OUTPUT_DIR,
    DEFAULT_VLM_QUERY_TYPE,
    VLMConfig,
    VLMFrameCropperLayerPackage,
    build_vlm_ack_package_from_result,
    build_vlm_output_json,
    format_vlm_output_json,
    initialize_vlm_layer,
    run_vlm_inference,
    save_vlm_debug_image,
)
from tracker import assign_tracking_status, build_tracking_layer_package, initialize_tracking_layer, update_tracks
from vlm_frame_cropper_layer import (
    build_vlm_dispatch_package,
    build_vlm_frame_cropper_package,
    build_vlm_frame_cropper_request_package,
    extract_vlm_object_crop,
    initialize_vlm_crop_cache,
    refresh_vlm_crop_cache_track_state,
    update_vlm_crop_cache,
)


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


def load_runtime_settings(config_path: Path | None = None) -> dict[str, Any]:
    config_path = config_path or (_CONFIG_DIR / "config.yaml")
    config = load_config(config_path)
    validate_config(config)
    return {
        "input_source": get_config_value(config, "config_input_source"),
        "input_path": _resolve_repo_path(get_config_value(config, "config_input_path")),
        "frame_resolution": tuple(get_config_value(config, "config_frame_resolution")),
        "yolo_model": get_config_value(config, "config_yolo_model"),
        "yolo_confidence_threshold": get_config_value(config, "config_yolo_confidence_threshold"),
        "device": get_config_value(config, "config_device"),
        "vlm_device": str(get_config_value(config, "config_vlm_device") or "").strip(),
        "vlm_enabled": bool(get_config_value(config, "config_vlm_enabled")),
        "vlm_backend": str(get_config_value(config, "config_vlm_backend")),
        "vlm_model": _resolve_vlm_model_path(get_config_value(config, "config_vlm_model")),
        "vlm_crop_feedback_enabled": bool(get_config_value(config, "config_vlm_crop_feedback_enabled")),
        "vlm_crop_cache_size": int(get_config_value(config, "config_vlm_crop_cache_size")),
        "vlm_dead_after_lost_frames": int(get_config_value(config, "config_vlm_dead_after_lost_frames")),
    }


def input_package_to_dict(package: Any) -> dict[str, Any]:
    return {
        "input_layer_frame_id": package.input_layer_frame_id,
        "input_layer_timestamp": package.input_layer_timestamp,
        "input_layer_image": package.input_layer_image,
        "input_layer_source_type": package.input_layer_source_type,
        "input_layer_resolution": package.input_layer_resolution,
    }


def tracking_rows(tracking_pkg: dict[str, Any]) -> list[dict[str, Any]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real config-driven VLM pipeline and save actual VLM debug images.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_DIR / "config.yaml",
        help="Path to configuration YAML (default: src/configuration-layer/config.yaml).",
    )
    parser.add_argument("--max-frames", type=int, default=5000, help="Maximum number of frames to scan before giving up.")
    parser.add_argument("--target-ids", type=int, default=15, help="Number of unique VLM-dispatched track IDs to capture.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--gstreamer", action="store_true")
    parser.add_argument("--device-override", default="", help="Override config_device for YOLO and VLM.")
    parser.add_argument("--vlm-device-override", default="", help="Override device only for VLM.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_VLM_DEBUG_OUTPUT_DIR)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    settings = load_runtime_settings(config_path=config_path)
    if not settings["vlm_enabled"]:
        raise RuntimeError("config_vlm_enabled is false in config.yaml, so the VLM path is disabled.")

    runtime_device = args.device_override.strip() or settings["device"]
    runtime_vlm_device = args.vlm_device_override.strip() or settings["vlm_device"] or runtime_device

    input_layer = InputLayer()
    input_layer.initialize_input_layer(
        config_input_source=settings["input_source"],
        config_frame_resolution=settings["frame_resolution"],
        config_input_path=settings["input_path"],
        camera_device_index=args.camera_index,
        use_gstreamer=args.gstreamer,
    )
    initialize_yolo_layer(
        model_name=settings["yolo_model"],
        conf_threshold=settings["yolo_confidence_threshold"],
        device=runtime_device,
    )
    initialize_tracking_layer(frame_rate=30)
    crop_cache_state = initialize_vlm_crop_cache(
        settings["vlm_crop_cache_size"],
        settings["vlm_dead_after_lost_frames"],
    )
    vlm_state = initialize_vlm_layer(
        VLMConfig(
            config_vlm_enabled=True,
            config_vlm_backend=settings["vlm_backend"],
            config_vlm_model=settings["vlm_model"],
            config_device=runtime_vlm_device,
        )
    )

    captured_track_ids: set[str] = set()
    saved_results: list[dict[str, Any]] = []

    try:
        for _ in range(args.max_frames):
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

            for index, row in enumerate(tracking_rows(tracking_pkg)):
                refresh_vlm_crop_cache_track_state(crop_cache_state, row, input_package.input_layer_frame_id)
                if row["status"] == "lost":
                    continue
                request_pkg = build_vlm_frame_cropper_request_package(
                    input_layer_package=input_package,
                    tracking_layer_package=tracking_pkg,
                    track_index=index,
                    vlm_frame_cropper_trigger_reason=f'tracking_status:{row["status"]}',
                    config_vlm_enabled=True,
                )
                if request_pkg is None:
                    continue
                crop = extract_vlm_object_crop(input_package, request_pkg)
                crop_pkg = build_vlm_frame_cropper_package(request_pkg, crop)
                update_vlm_crop_cache(
                    crop_cache_state,
                    row,
                    crop_pkg,
                    input_package.input_layer_frame_id,
                    f'tracking_status:{row["status"]}',
                )

            for tid in sorted(crop_cache_state["track_caches"].keys(), key=lambda value: (len(str(value)), str(value))):
                if str(tid) in captured_track_ids:
                    continue
                dispatch = build_vlm_dispatch_package(crop_cache_state, tid)
                if dispatch is None:
                    continue

                inner = dispatch["vlm_frame_cropper_layer_package"]
                vlm_crop_pkg = VLMFrameCropperLayerPackage(
                    vlm_frame_cropper_layer_track_id=str(inner["vlm_frame_cropper_layer_track_id"]),
                    vlm_frame_cropper_layer_image=inner["vlm_frame_cropper_layer_image"],
                    vlm_frame_cropper_layer_bbox=tuple(int(x) for x in inner["vlm_frame_cropper_layer_bbox"]),
                )

                query_type = DEFAULT_VLM_QUERY_TYPE if settings["vlm_crop_feedback_enabled"] else "vehicle_semantics_single_shot_v1"
                if dispatch["vlm_dispatch_mode"] == "dead_best_available":
                    query_type = "vehicle_semantics_single_shot_v1"

                raw_result = run_vlm_inference(
                    vlm_runtime_state=vlm_state,
                    vlm_frame_cropper_layer_package=vlm_crop_pkg,
                    vlm_layer_query_type=query_type,
                )
                ack_package = build_vlm_ack_package_from_result(raw_result)
                image_path = save_vlm_debug_image(
                    vlm_frame_cropper_layer_package=vlm_crop_pkg,
                    vlm_layer_raw_result=raw_result,
                    output_dir=args.output_dir,
                    file_stem=(
                        f'frame_{int(input_package.input_layer_frame_id):05d}'
                        f'__track_{str(inner["vlm_frame_cropper_layer_track_id"])}'
                        f'__{dispatch["vlm_dispatch_mode"]}'
                    ),
                )

                result_payload = build_vlm_output_json(raw_result)
                summary = {
                    "frame_id": int(input_package.input_layer_frame_id),
                    "dispatch_track_id": dispatch["vlm_dispatch_track_id"],
                    "dispatch_mode": dispatch["vlm_dispatch_mode"],
                    "dispatch_reason": dispatch["vlm_dispatch_reason"],
                    "saved_debug_image": str(image_path),
                    "ack_status": ack_package.vlm_ack_status,
                    "ack_reason": ack_package.vlm_ack_reason,
                    "vlm_raw_text": raw_result.vlm_layer_raw_text,
                    "vlm_output_json": result_payload,
                }
                captured_track_ids.add(str(tid))
                saved_results.append(summary)
                print(
                    f'Captured {len(saved_results)} / {args.target_ids}: '
                    f'track {dispatch["vlm_dispatch_track_id"]} '
                    f'-> {image_path}'
                )
                if len(saved_results) >= args.target_ids:
                    print("Real config-driven VLM run complete.")
                    print(json.dumps(saved_results, indent=2))
                    return

        raise RuntimeError(
            f"Captured {len(saved_results)} unique VLM-dispatched track IDs within {args.max_frames} frames, "
            f"but target was {args.target_ids}."
        )
    finally:
        input_layer.close_input_layer()


if __name__ == "__main__":
    main()
