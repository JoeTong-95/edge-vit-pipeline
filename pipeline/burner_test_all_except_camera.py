#!/usr/bin/env python3
"""
Burner test (all-except-camera).

Goal: validate layer-wise public functionality without using live camera input.
This script is intentionally lightweight and avoids GPU/VLM model loading.

What it exercises (best-effort, in order):
- configuration-layer: load_config / validate_config / get_config_value
- input-layer: video ingestion + input_layer_package creation (one frame)
- roi-layer: init/apply/build package + lock logic using detections
- yolo-layer: init/detect/filter/package (one frame)
- tracking-layer: init/update/status/package (one frame)
- vehicle-state-layer: init/update from tracking + build package
- vlm-frame-cropper-layer: request/crop/package + cache/dispatch + ack registration
- vlm-layer: prompt + parsing + normalization + ack building (no model inference)
- scene-awareness-layer (if present): init + inference stub + package shape
- evaluation-output-layer (if present): metrics + package shape + stdout emission
- metadata-output-layer (if present): package shape + deterministic JSON serialization

Run from repo root:
  python pipeline/burner_test_all_except_camera.py

This script intentionally has **no CLI overrides**. It reads
`src/configuration-layer/config.yaml` and exercises the configured pipeline path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

CONFIG_DIR = SRC_DIR / "configuration-layer"
INPUT_DIR = SRC_DIR / "input-layer"
ROI_DIR = SRC_DIR / "roi-layer"
YOLO_DIR = SRC_DIR / "yolo-layer"
TRACKING_DIR = SRC_DIR / "tracking-layer"
VSTATE_DIR = SRC_DIR / "vehicle-state-layer"
CROPPER_DIR = SRC_DIR / "vlm-frame-cropper-layer"
VLM_DIR = SRC_DIR / "vlm-layer"

SCENE_DIR = SRC_DIR / "scene-awareness-layer"
EVAL_DIR = SRC_DIR / "evaluation-output-layer"
META_DIR = SRC_DIR / "metadata-output-layer"


def _add_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


for p in (
    CONFIG_DIR,
    INPUT_DIR,
    ROI_DIR,
    YOLO_DIR,
    TRACKING_DIR,
    VSTATE_DIR,
    CROPPER_DIR,
    VLM_DIR,
    SCENE_DIR,
    EVAL_DIR,
    META_DIR,
):
    if p.exists():
        _add_sys_path(p)


def _require(module_name: str) -> Any:
    try:
        return __import__(module_name)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Required dependency '{module_name}' is not available: {exc}") from exc


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _pretty(msg: str) -> None:
    print(f"[burner] {msg}")


def _resolve_video_path(cli_video: str | None, config_video: str | None) -> str:
    raw = (cli_video or config_video or "").strip()
    if not raw:
        raise RuntimeError("No video path provided (use --video or set config_input_path).")
    path = Path(raw)
    if not path.is_absolute():
        path = (REPO_ROOT / raw).resolve()
    return str(path)


def main() -> None:
    max_frames = 3

    # --- Dependencies needed for the implemented pipeline path ---
    _require("cv2")
    _require("numpy")

    # --- Configuration layer ---
    _pretty("config: load/validate")
    from config_node import get_config_value, load_config, validate_config

    config_path = CONFIG_DIR / "config.yaml"
    cfg = load_config(config_path)
    validate_config(cfg)

    config_input_path = get_config_value(cfg, "config_input_path")
    config_frame_resolution = tuple(get_config_value(cfg, "config_frame_resolution"))
    config_device = get_config_value(cfg, "config_device")
    config_yolo_model = get_config_value(cfg, "config_yolo_model")
    config_yolo_conf = float(get_config_value(cfg, "config_yolo_confidence_threshold"))
    config_roi_enabled = bool(get_config_value(cfg, "config_roi_enabled"))
    config_roi_thresh = int(get_config_value(cfg, "config_roi_vehicle_count_threshold"))

    video_path = _resolve_video_path(None, str(config_input_path))
    _assert(os.path.exists(video_path), f"Video does not exist: {video_path}")

    # --- Input layer (video) ---
    _pretty("input: read video frame(s)")
    from input_layer import InputLayer

    input_layer = InputLayer()
    input_layer.initialize_input_layer(
        config_input_source="video",
        config_frame_resolution=config_frame_resolution,
        config_input_path=video_path,
        camera_device_index=0,
        use_gstreamer=False,
    )
    raw_frame = input_layer.read_next_frame()
    _assert(raw_frame is not None, "Failed to read first video frame.")
    input_pkg_obj = input_layer.build_input_layer_package(raw_frame)
    input_pkg = {
        "input_layer_frame_id": input_pkg_obj.input_layer_frame_id,
        "input_layer_timestamp": input_pkg_obj.input_layer_timestamp,
        "input_layer_image": input_pkg_obj.input_layer_image,
        "input_layer_source_type": input_pkg_obj.input_layer_source_type,
        "input_layer_resolution": input_pkg_obj.input_layer_resolution,
    }
    _assert("input_layer_image" in input_pkg, "input_layer_package missing image.")

    # --- ROI layer ---
    _pretty("roi: init/apply/build")
    from roi_layer import build_roi_layer_package, initialize_roi_layer, lock_roi_bounds, update_roi_state

    initialize_roi_layer(config_roi_enabled=config_roi_enabled, config_roi_vehicle_count_threshold=config_roi_thresh)
    roi_pkg = build_roi_layer_package(input_pkg)
    _assert("roi_layer_image" in roi_pkg, "roi_layer_package missing image.")
    _assert("roi_layer_bounds" in roi_pkg, "roi_layer_package missing bounds.")

    # Force-lock ROI via synthetic detections (avoids needing many frames).
    if config_roi_enabled:
        frame_h, frame_w = input_pkg["input_layer_image"].shape[:2]
        fake_dets = [
            {"yolo_detection_bbox": (frame_w * 0.1, frame_h * 0.2, frame_w * 0.6, frame_h * 0.9), "yolo_detection_class": "truck", "yolo_detection_confidence": 0.9},
            {"yolo_detection_bbox": (frame_w * 0.15, frame_h * 0.25, frame_w * 0.55, frame_h * 0.85), "yolo_detection_class": "truck", "yolo_detection_confidence": 0.8},
        ]
        _ = update_roi_state(input_pkg, fake_dets)
        # Ensure lock for downstream ROI path tests.
        lock_roi_bounds((0, 0, frame_w, frame_h))
        roi_pkg = build_roi_layer_package(input_pkg)
        _assert(bool(roi_pkg.get("roi_layer_locked")) is True, "ROI should be locked after lock_roi_bounds.")

    # --- YOLO layer ---
    _pretty("yolo: init/detect/filter/package")
    _require("ultralytics")
    from detector import build_yolo_layer_package, filter_yolo_detections, initialize_yolo_layer, run_yolo_detection

    initialize_yolo_layer(
        model_name=config_yolo_model,
        conf_threshold=config_yolo_conf,
        device=str(config_device or "").strip(),
    )
    upstream_for_yolo = roi_pkg if (config_roi_enabled and roi_pkg.get("roi_layer_locked")) else input_pkg
    raw_dets = run_yolo_detection(upstream_for_yolo)
    filtered = filter_yolo_detections(raw_dets)
    yolo_pkg = build_yolo_layer_package(
        upstream_for_yolo.get("roi_layer_frame_id", upstream_for_yolo.get("input_layer_frame_id")),
        filtered,
    )
    _assert("yolo_layer_detections" in yolo_pkg, "yolo_layer_package missing detections.")

    # --- Tracking layer ---
    _pretty("tracking: init/update/status/package")
    _require("supervision")
    from tracker import assign_tracking_status, build_tracking_layer_package, initialize_tracking_layer, update_tracks

    initialize_tracking_layer(frame_rate=30)
    tracking_pkg = None
    for _ in range(max_frames):
        current_tracks = update_tracks(yolo_pkg)
        status_tracks = assign_tracking_status(current_tracks, yolo_pkg["yolo_layer_frame_id"])
        tracking_pkg = build_tracking_layer_package(yolo_pkg["yolo_layer_frame_id"], status_tracks)

        raw_frame = input_layer.read_next_frame()
        if raw_frame is None:
            break
        input_pkg_obj = input_layer.build_input_layer_package(raw_frame)
        input_pkg = {
            "input_layer_frame_id": input_pkg_obj.input_layer_frame_id,
            "input_layer_timestamp": input_pkg_obj.input_layer_timestamp,
            "input_layer_image": input_pkg_obj.input_layer_image,
            "input_layer_source_type": input_pkg_obj.input_layer_source_type,
            "input_layer_resolution": input_pkg_obj.input_layer_resolution,
        }
        if config_roi_enabled:
            roi_pkg = build_roi_layer_package(input_pkg)
        upstream_for_yolo = roi_pkg if (config_roi_enabled and roi_pkg.get("roi_layer_locked")) else input_pkg
        raw_dets = run_yolo_detection(upstream_for_yolo)
        filtered = filter_yolo_detections(raw_dets)
        yolo_pkg = build_yolo_layer_package(
            upstream_for_yolo.get("roi_layer_frame_id", upstream_for_yolo.get("input_layer_frame_id")),
            filtered,
        )
    _assert(isinstance(tracking_pkg, dict), "tracking layer did not produce a tracking package.")
    _assert("tracking_layer_track_id" in tracking_pkg, "tracking_layer_package missing track ids.")

    # --- Vehicle state layer ---
    _pretty("vehicle_state: init/update/build")
    from vehicle_state_layer import (
        build_vehicle_state_layer_package,
        initialize_vehicle_state_layer,
        update_vehicle_state_from_tracking,
    )

    initialize_vehicle_state_layer(prune_after_lost_frames=None)
    update_vehicle_state_from_tracking(tracking_pkg)
    vs_pkg = build_vehicle_state_layer_package()
    _assert("vehicle_state_layer_track_id" in vs_pkg, "vehicle_state_layer_package missing track ids.")

    # --- Cropper layer ---
    _pretty("cropper: request/crop/cache/dispatch/ack")
    from vlm_frame_cropper_layer import (
        build_vlm_dispatch_package,
        build_vlm_frame_cropper_package,
        build_vlm_frame_cropper_request_package,
        extract_vlm_object_crop,
        initialize_vlm_crop_cache,
        register_vlm_ack_package,
        update_vlm_crop_cache,
    )

    crop_cache = initialize_vlm_crop_cache(config_vlm_crop_cache_size=3, config_vlm_dead_after_lost_frames=2)

    # If no tracks exist (e.g. empty frame), fabricate one tracking row for cropper validation.
    if not tracking_pkg["tracking_layer_track_id"]:
        tracking_pkg = {
            "tracking_layer_frame_id": input_pkg["input_layer_frame_id"],
            "tracking_layer_track_id": ["1"],
            "tracking_layer_bbox": [(10, 10, 60, 60)],
            "tracking_layer_detector_class": ["truck"],
            "tracking_layer_confidence": [0.9],
            "tracking_layer_status": ["active"],
        }

    # Build one request + crop + cache-fill to force a dispatch.
    for round_idx in range(3):
        req = build_vlm_frame_cropper_request_package(
            input_layer_package=input_pkg,
            tracking_layer_package=tracking_pkg,
            track_index=0,
            vlm_frame_cropper_trigger_reason=f"burner_round:{round_idx}",
            config_vlm_enabled=True,
        )
        _assert(req is not None, "Expected cropper request package.")
        crop = extract_vlm_object_crop(input_pkg, req)
        crop_pkg = build_vlm_frame_cropper_package(req, crop)
        row = {
            "track_id": str(tracking_pkg["tracking_layer_track_id"][0]),
            "bbox": tuple(tracking_pkg["tracking_layer_bbox"][0]),
            "detector_class": str(tracking_pkg["tracking_layer_detector_class"][0]),
            "confidence": float(tracking_pkg["tracking_layer_confidence"][0]),
            "status": str(tracking_pkg["tracking_layer_status"][0]),
        }
        update_vlm_crop_cache(crop_cache, row, crop_pkg, int(input_pkg["input_layer_frame_id"]), "burner")

    dispatch = build_vlm_dispatch_package(crop_cache, str(tracking_pkg["tracking_layer_track_id"][0]))
    _assert(dispatch is not None, "Expected a dispatch after cache fill.")
    inner = dispatch["vlm_frame_cropper_layer_package"]
    _assert("vlm_frame_cropper_layer_image" in inner, "Dispatch inner package missing image.")

    # --- VLM layer (no model load) ---
    _pretty("vlm: prompt/parse/normalize/ack (no model)")
    from layer import (
        VLMFrameCropperLayerPackage,
        VLMRawResult,
        build_vlm_ack_package_from_result,
        normalize_vlm_result,
        parse_vlm_response,
        prepare_vlm_prompt,
    )

    vlm_crop_pkg = VLMFrameCropperLayerPackage(
        vlm_frame_cropper_layer_track_id=str(inner["vlm_frame_cropper_layer_track_id"]),
        vlm_frame_cropper_layer_image=inner["vlm_frame_cropper_layer_image"],
        vlm_frame_cropper_layer_bbox=tuple(inner["vlm_frame_cropper_layer_bbox"]),
    )
    prompt = prepare_vlm_prompt("vehicle_semantics_single_shot_v1", vlm_crop_pkg)
    _assert("Allowed" in prompt, "VLM prompt did not include allowed-label hint.")

    parsed = parse_vlm_response('{"is_truck":true,"wheel_count":18,"estimated_weight_kg":24000,"ack_status":"accepted","retry_reasons":[]}')
    _assert(parsed.get("wheel_count") == 18, "VLM parse failed wheel_count.")
    raw = VLMRawResult(
        vlm_layer_track_id=str(inner["vlm_frame_cropper_layer_track_id"]),
        vlm_layer_query_type="vehicle_semantics_single_shot_v1",
        vlm_layer_model_id="burner",
        vlm_layer_raw_text='{"is_truck":true,"wheel_count":18,"estimated_weight_kg":24000,"ack_status":"accepted","retry_reasons":[]}',
        vlm_layer_raw_response={"prompt_text": prompt},
    )
    normalized = normalize_vlm_result(raw)
    _assert(normalized.get("vlm_ack_status") == "accepted", "VLM normalize did not produce accepted ack.")
    ack = build_vlm_ack_package_from_result(raw)
    _assert(ack.vlm_ack_status == "accepted", "VLM ack package not accepted in single-shot.")

    # Register ack back into cropper (ensures ack loop wiring works).
    register_vlm_ack_package(crop_cache, ack)

    # --- Scene awareness layer (optional) ---
    scene_pkg = None
    if (SCENE_DIR / "scene_awareness_layer.py").is_file():
        _pretty("scene_awareness: init/infer (stub)")
        from scene_awareness_layer import initialize_scene_awareness_layer, run_scene_awareness_inference

        scene_state = initialize_scene_awareness_layer(
            config_scene_awareness_enabled=True, config_device="auto"
        )
        scene_pkg = run_scene_awareness_inference(scene_state, input_pkg)
        _assert(scene_pkg is not None, "Scene awareness inference returned None while enabled.")
        for k in (
            "scene_awareness_layer_frame_id",
            "scene_awareness_layer_timestamp",
            "scene_awareness_layer_label",
            "scene_awareness_layer_attributes",
            "scene_awareness_layer_confidence",
        ):
            _assert(k in scene_pkg, f"Scene awareness package missing {k}.")

    # --- Evaluation output layer (optional) ---
    if (EVAL_DIR / "evaluation_output_layer.py").is_file():
        _pretty("evaluation_output: collect/build/emit stdout")
        from evaluation_output_layer import build_evaluation_output_layer_package, collect_evaluation_metrics, emit_evaluation_output

        metrics = collect_evaluation_metrics(
            input_layer_package=input_pkg,
            roi_layer_package=roi_pkg,
            yolo_layer_package=yolo_pkg,
            tracking_layer_package=tracking_pkg,
            vlm_layer_package=None,
            scene_awareness_layer_package=scene_pkg,
            timings={"total_ms": 50.0},
        )
        eval_pkg = build_evaluation_output_layer_package(metrics)
        for k in (
            "evaluation_output_layer_fps",
            "evaluation_output_layer_module_latency",
            "evaluation_output_layer_detection_count",
            "evaluation_output_layer_track_count",
            "evaluation_output_layer_vlm_call_count",
            "evaluation_output_layer_scene_call_count",
        ):
            _assert(k in eval_pkg, f"Evaluation output package missing {k}.")
        emit_evaluation_output(eval_pkg, output_destination="stdout")

    # --- Metadata output layer (optional) ---
    if (META_DIR / "metadata_output_layer.py").is_file():
        _pretty("metadata_output: build/serialize stdout")
        from metadata_output_layer import build_metadata_output_layer_package, emit_metadata_output, serialize_metadata_output

        meta_pkg = build_metadata_output_layer_package(
            vehicle_state_layer_package=vs_pkg,
            vlm_layer_package=None,
            scene_awareness_layer_package=scene_pkg,
        )
        for k in (
            "metadata_output_layer_timestamps",
            "metadata_output_layer_object_ids",
            "metadata_output_layer_classes",
            "metadata_output_layer_semantic_tags",
            "metadata_output_layer_scene_tags",
            "metadata_output_layer_counts",
            "metadata_output_layer_summaries",
        ):
            _assert(k in meta_pkg, f"Metadata output package missing {k}.")
        payload = serialize_metadata_output(meta_pkg, output_format="json")
        emit_metadata_output(payload, output_destination="stdout")

    input_layer.close_input_layer()
    _pretty("PASS ✅")


if __name__ == "__main__":
    main()

