#!/usr/bin/env python3

"""
initialize_pipeline.py (repo root)

Loads `src/configuration-layer/config.yaml` and runs the full video pipeline:
input → ROI → YOLO → tracking → vehicle state → VLM cropper/dispatch (when enabled),
optionally scene awareness.

VLM outputs are appended as **JSON Lines** (default `data/pipeline_output.jsonl`) as a
local stand-in for TTN uplink — one JSON object per completed VLM result (inline batch
or async worker drain).

Usage:
  python initialize_pipeline.py
  python initialize_pipeline.py --max-frames 500 --output data/pipeline_output.jsonl
"""

from __future__ import annotations

import os as _os_early
if "PYTORCH_CUDA_ALLOC_CONF" not in _os_early.environ:
    _os_early.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
del _os_early

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"

CONFIG_DIR = SRC_DIR / "configuration-layer"
INPUT_DIR = SRC_DIR / "input-layer"
ROI_DIR = SRC_DIR / "roi-layer"
YOLO_DIR = SRC_DIR / "yolo-layer"
TRACKING_DIR = SRC_DIR / "tracking-layer"
VSTATE_DIR = SRC_DIR / "vehicle-state-layer"
SCENE_DIR = SRC_DIR / "scene-awareness-layer"
VLM_DIR = SRC_DIR / "vlm-layer"
VLM_UTIL_DIR = SRC_DIR / "vlm-layer" / "util"
CROPPER_DIR = SRC_DIR / "vlm-frame-cropper-layer"

for p in (
    CONFIG_DIR,
    INPUT_DIR,
    ROI_DIR,
    YOLO_DIR,
    TRACKING_DIR,
    VSTATE_DIR,
    SCENE_DIR,
    VLM_DIR,
    VLM_UTIL_DIR,
    CROPPER_DIR,
):
    if p.exists():
        sys.path.insert(0, str(p))


def _resolve_repo_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _append_jsonl(fh: Any, record: dict[str, Any]) -> None:
    fh.write(json.dumps(record, default=str) + "\n")
    fh.flush()


def _dispatch_subset(dispatch: dict[str, Any]) -> dict[str, Any]:
    return {
        "vlm_dispatch_track_id": dispatch.get("vlm_dispatch_track_id"),
        "vlm_dispatch_mode": dispatch.get("vlm_dispatch_mode"),
        "vlm_dispatch_reason": dispatch.get("vlm_dispatch_reason"),
        "vlm_dispatch_cached_crop_count": dispatch.get("vlm_dispatch_cached_crop_count"),
    }


def main() -> None:
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401

    parser = argparse.ArgumentParser(description="Run full pipeline; log VLM output to JSONL (TTN substitute).")
    parser.add_argument(
        "--output",
        default="data/pipeline_output.jsonl",
        help="JSON Lines file for VLM records (default: data/pipeline_output.jsonl).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the output file instead of truncating at start.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames (0 = until video ends).",
    )
    parser.add_argument(
        "--video",
        default="",
        help="Override config_input_path (repo-relative or absolute).",
    )
    parser.add_argument(
        "--include-raw-vlm",
        action="store_true",
        help="Include vlm_raw_result blobs in each record (larger files).",
    )
    args = parser.parse_args()

    from config_node import get_config_value, load_config, validate_config
    from input_layer import InputLayer
    from roi_layer import build_roi_layer_package, initialize_roi_layer, update_roi_state
    from detector import (
        build_yolo_layer_package,
        filter_yolo_detections,
        initialize_yolo_layer,
        run_yolo_detection,
    )
    from tracker import (
        assign_tracking_status,
        build_tracking_layer_package,
        initialize_tracking_layer,
        update_tracks,
    )
    from vehicle_state_layer import initialize_vehicle_state_layer, update_vehicle_state_from_tracking

    scene_available = (SCENE_DIR / "scene_awareness_layer.py").is_file()
    if scene_available:
        from scene_awareness_layer import initialize_scene_awareness_layer, run_scene_awareness_inference
    else:
        initialize_scene_awareness_layer = None  # type: ignore
        run_scene_awareness_inference = None  # type: ignore

    config_path = CONFIG_DIR / "config.yaml"
    cfg = load_config(config_path)
    validate_config(cfg)

    input_source = str(get_config_value(cfg, "config_input_source"))
    if input_source not in ("video", "camera"):
        raise RuntimeError("initialize_pipeline.py supports video and camera only.")

    configured_video_path = str(get_config_value(cfg, "config_input_path"))
    video_path_value = args.video.strip() or configured_video_path
    video_path = _resolve_repo_path(str(video_path_value))
    if input_source == "video" and not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path not found: {video_path}")

    frame_resolution = tuple(get_config_value(cfg, "config_frame_resolution"))
    device = str(get_config_value(cfg, "config_device"))
    model = str(get_config_value(cfg, "config_yolo_model"))
    conf = float(get_config_value(cfg, "config_yolo_confidence_threshold"))
    roi_enabled = bool(get_config_value(cfg, "config_roi_enabled"))
    roi_threshold = int(get_config_value(cfg, "config_roi_vehicle_count_threshold"))
    scene_enabled = bool(get_config_value(cfg, "config_scene_awareness_enabled"))

    vlm_enabled = bool(get_config_value(cfg, "config_vlm_enabled"))
    vlm_model = str(get_config_value(cfg, "config_vlm_model"))
    vlm_crop_cache_size = int(get_config_value(cfg, "config_vlm_crop_cache_size"))
    vlm_dead_after_lost_frames = int(get_config_value(cfg, "config_vlm_dead_after_lost_frames"))
    vlm_worker_batch_wait_ms = int(get_config_value(cfg, "config_vlm_worker_batch_wait_ms"))
    vlm_worker_max_queue_size = int(get_config_value(cfg, "config_vlm_worker_max_queue_size"))
    vlm_runtime_mode_cfg = str(get_config_value(cfg, "config_vlm_runtime_mode"))
    vlm_batch_size = max(1, int(get_config_value(cfg, "config_vlm_worker_batch_size")))

    vlm_bench_mode = str(vlm_runtime_mode_cfg or "inline").strip().lower()
    if vlm_bench_mode not in ("inline", "async", "spill"):
        vlm_bench_mode = "inline"
    vlm_runtime_mode_intent = vlm_bench_mode

    out_path = Path(_resolve_repo_path(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    input_layer = InputLayer()
    if input_source == "camera":
        input_layer.initialize_input_layer(
            config_input_source="camera",
            config_frame_resolution=frame_resolution,
            camera_device_index=0,
        )
    else:
        input_layer.initialize_input_layer(
            config_input_source="video",
            config_frame_resolution=frame_resolution,
            config_input_path=video_path,
        )

    initialize_roi_layer(config_roi_enabled=roi_enabled, config_roi_vehicle_count_threshold=roi_threshold)
    initialize_yolo_layer(model_name=model, conf_threshold=conf, device=device)
    initialize_tracking_layer(frame_rate=30)
    initialize_vehicle_state_layer(prune_after_lost_frames=None)

    vlm_available = False
    vlm_state = None
    crop_cache = None
    vlm_worker = None
    vlm_calls = 0
    vlm_query_type = "vehicle_semantics_single_shot_v1"
    vlm_effective_runtime = "inline"

    build_vlm_output_json = None
    VLMFrameCropperLayerPackage = None

    if vlm_enabled:
        try:
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
            from layer import (
                VLMConfig,
                VLMFrameCropperLayerPackage as _VLMFrameCropperLayerPackage,
                build_vlm_ack_package_from_result,
                build_vlm_output_json as _build_vlm_output_json,
                initialize_vlm_layer,
                prepare_vlm_prompt,
                run_vlm_inference_batch,
            )

            VLMFrameCropperLayerPackage = _VLMFrameCropperLayerPackage
            build_vlm_output_json = _build_vlm_output_json

            vlm_state = initialize_vlm_layer(
                VLMConfig(
                    config_vlm_enabled=True,
                    config_vlm_model=_resolve_repo_path(vlm_model),
                    config_device=device,
                )
            )
            crop_cache = initialize_vlm_crop_cache(
                config_vlm_crop_cache_size=vlm_crop_cache_size,
                config_vlm_dead_after_lost_frames=vlm_dead_after_lost_frames,
            )
            vlm_available = True
            if vlm_available and vlm_bench_mode in ("async", "spill"):
                try:
                    from visualize_vlm_realtime import AsyncVLMWorker

                    spill_path = ""
                    spill_max_bytes = 0
                    if vlm_bench_mode == "spill":
                        spill_raw = str(get_config_value(cfg, "config_vlm_worker_spill_queue_path")).strip()
                        if spill_raw:
                            spill_path = _resolve_repo_path(spill_raw)
                        spill_mb = float(get_config_value(cfg, "config_vlm_spill_max_file_mb"))
                        spill_max_bytes = int(spill_mb * 1024 * 1024) if spill_mb > 0 else 0
                    vlm_crop_feedback_enabled = bool(get_config_value(cfg, "config_vlm_crop_feedback_enabled"))
                    vlm_worker = AsyncVLMWorker(
                        vlm_state=vlm_state,
                        feedback_enabled=vlm_crop_feedback_enabled,
                        max_queue_size=vlm_worker_max_queue_size,
                        batch_size=vlm_batch_size,
                        batch_wait_ms=vlm_worker_batch_wait_ms,
                        spill_queue_path=spill_path,
                        spill_max_file_bytes=spill_max_bytes,
                    )
                    vlm_worker.start()
                except Exception as exc:
                    print(f"[pipeline] Async VLM worker failed ({type(exc).__name__}: {exc}); using inline VLM.")
                    vlm_worker = None
        except Exception as exc:
            print(f"[pipeline] VLM unavailable ({type(exc).__name__}: {exc}); continuing without VLM.")
            vlm_available = False

    vlm_effective_runtime = vlm_runtime_mode_intent
    if vlm_enabled and vlm_available and vlm_runtime_mode_intent in ("async", "spill") and vlm_worker is None:
        vlm_effective_runtime = "inline"

    scene_state = None
    if scene_enabled and scene_available and initialize_scene_awareness_layer is not None:
        scene_state = initialize_scene_awareness_layer(config_scene_awareness_enabled=True, config_device=device)

    lines_written = 0
    frames_done = 0
    last_frame_id = -1

    def log_worker_results(fh: Any, results: list[dict[str, Any]], current_frame_id: int) -> None:
        nonlocal lines_written
        for r in results:
            ack = r.get("ack")
            raw_result = r.get("raw_result")
            rec: dict[str, Any] = {
                "schema": "pipeline_output_v1",
                "logged_at_unix_s": time.time(),
                "input_frame_id": int(current_frame_id),
                "vlm_runtime_mode": vlm_effective_runtime,
                "dispatch_frame_id": int(r.get("dispatch_frame_id", -1)),
                "track_id": str(r.get("track_id", "")),
                "query_type": str(r.get("query_type", "")),
                "worker_runtime_sec": float(r.get("runtime_sec", 0.0)),
                "error_text": str(r.get("error_text") or ""),
            }
            if ack is not None:
                try:
                    rec["vlm_ack"] = asdict(ack)
                except Exception:
                    rec["vlm_ack"] = str(ack)
            if raw_result is not None and build_vlm_output_json is not None:
                rec["vlm"] = build_vlm_output_json(raw_result, include_raw_result=bool(args.include_raw_vlm))
            else:
                rec["vlm"] = None
            _append_jsonl(fh, rec)
            lines_written += 1

    def drain_vlm_worker(fh: Any, current_frame_id: int) -> None:
        nonlocal vlm_calls
        if vlm_worker is None or crop_cache is None:
            return
        batch_results = vlm_worker.drain_results()
        if not batch_results:
            return
        vlm_calls += len(batch_results)
        for br in batch_results:
            register_vlm_ack_package(crop_cache, br["ack"])
        log_worker_results(fh, batch_results, current_frame_id)

    with open(out_path, mode, encoding="utf-8") as out_f:
        max_frames = int(args.max_frames)
        _pipeline_start_time = time.time()
        while True:
            if max_frames > 0 and frames_done >= max_frames:
                break

            raw = input_layer.read_next_frame()
            if raw is None:
                if input_source == "camera":
                    time.sleep(0.01)  # camera dropped a frame, retry
                    continue
                break  # video ended

            # Progress logging every 100 frames
            if frames_done > 0 and frames_done % 100 == 0:
                elapsed = time.time() - _pipeline_start_time
                fps = frames_done / elapsed if elapsed > 0 else 0
                vlm_status = ""
                if vlm_worker is not None:
                    st = vlm_worker.get_status()
                    vlm_status = f" vlm_queue={st['queue_size']} vlm_done={st['completed_count']}"
                print(f"[pipeline] frame={frames_done} fps={fps:.1f} vlm_logged={lines_written}{vlm_status}")
            pkg = input_layer.build_input_layer_package(raw)
            input_pkg = {
                "input_layer_frame_id": pkg.input_layer_frame_id,
                "input_layer_timestamp": pkg.input_layer_timestamp,
                "input_layer_image": pkg.input_layer_image,
                "input_layer_source_type": pkg.input_layer_source_type,
                "input_layer_resolution": pkg.input_layer_resolution,
            }
            frame_id = int(input_pkg["input_layer_frame_id"])
            last_frame_id = frame_id

            roi_pkg = build_roi_layer_package(input_pkg)
            dets_boot = run_yolo_detection(roi_pkg if roi_pkg.get("roi_layer_locked") else input_pkg)
            dets_boot = filter_yolo_detections(dets_boot)
            _ = update_roi_state(input_pkg, dets_boot)
            roi_pkg = build_roi_layer_package(input_pkg)

            yolo_upstream = roi_pkg if (roi_enabled and roi_pkg.get("roi_layer_locked")) else input_pkg
            dets = run_yolo_detection(yolo_upstream)
            dets = filter_yolo_detections(dets)
            yolo_pkg = build_yolo_layer_package(input_pkg["input_layer_frame_id"], dets)

            current_tracks = update_tracks(yolo_pkg)
            status_tracks = assign_tracking_status(current_tracks, input_pkg["input_layer_frame_id"])
            tracking_pkg = build_tracking_layer_package(input_pkg["input_layer_frame_id"], status_tracks)

            update_vehicle_state_from_tracking(tracking_pkg)

            if (
                vlm_enabled
                and vlm_available
                and crop_cache is not None
                and vlm_state is not None
                and VLMFrameCropperLayerPackage is not None
                and build_vlm_output_json is not None
            ):
                track_ids = tracking_pkg.get("tracking_layer_track_id", [])
                bboxes = tracking_pkg.get("tracking_layer_bbox", [])
                classes = tracking_pkg.get("tracking_layer_detector_class", [])
                confidences = tracking_pkg.get("tracking_layer_confidence", [])
                statuses = tracking_pkg.get("tracking_layer_status", [])

                for i in range(len(track_ids)):
                    row = {
                        "track_id": str(track_ids[i]),
                        "bbox": tuple(bboxes[i]),
                        "detector_class": str(classes[i]),
                        "confidence": float(confidences[i]),
                        "status": str(statuses[i]),
                    }
                    refresh_vlm_crop_cache_track_state(crop_cache, row, input_pkg["input_layer_frame_id"])
                    if row["status"] == "lost":
                        continue

                    req = build_vlm_frame_cropper_request_package(
                        input_layer_package=input_pkg,
                        tracking_layer_package=tracking_pkg,
                        track_index=i,
                        vlm_frame_cropper_trigger_reason=f"profile_tracking_status:{row['status']}",
                        config_vlm_enabled=True,
                    )
                    if req is None:
                        continue
                    crop = extract_vlm_object_crop(input_pkg, req)
                    crop_pkg = build_vlm_frame_cropper_package(req, crop)
                    update_vlm_crop_cache(
                        crop_cache,
                        row,
                        crop_pkg,
                        int(input_pkg["input_layer_frame_id"]),
                        f"profile_tracking_status:{row['status']}",
                    )

                if vlm_effective_runtime in ("async", "spill") and vlm_worker is not None:
                    drain_vlm_worker(out_f, frame_id)
                    for tid in sorted(crop_cache["track_caches"].keys(), key=lambda x: (len(str(x)), str(x))):
                        dispatch = build_vlm_dispatch_package(crop_cache, tid)
                        if dispatch is None:
                            continue
                        inner = dispatch["vlm_frame_cropper_layer_package"]
                        vlm_crop_pkg = VLMFrameCropperLayerPackage(
                            vlm_frame_cropper_layer_track_id=str(inner["vlm_frame_cropper_layer_track_id"]),
                            vlm_frame_cropper_layer_image=inner["vlm_frame_cropper_layer_image"],
                            vlm_frame_cropper_layer_bbox=tuple(int(x) for x in inner["vlm_frame_cropper_layer_bbox"]),
                        )
                        vlm_worker.submit(
                            {
                                "track_id": str(inner["vlm_frame_cropper_layer_track_id"]),
                                "dispatch_frame_id": int(input_pkg["input_layer_frame_id"]),
                                "prompt_text": prepare_vlm_prompt(vlm_query_type, vlm_crop_pkg),
                                "query_type": vlm_query_type,
                                "submitted_at": time.time(),
                                "vlm_crop_pkg": vlm_crop_pkg,
                            }
                        )
                else:
                    dispatch_pkgs: list[Any] = []
                    dispatches: list[dict[str, Any]] = []
                    for tid in sorted(crop_cache["track_caches"].keys(), key=lambda x: (len(str(x)), str(x))):
                        dispatch = build_vlm_dispatch_package(crop_cache, tid)
                        if dispatch is None:
                            continue
                        inner = dispatch["vlm_frame_cropper_layer_package"]
                        dispatch_pkgs.append(
                            VLMFrameCropperLayerPackage(
                                vlm_frame_cropper_layer_track_id=str(inner["vlm_frame_cropper_layer_track_id"]),
                                vlm_frame_cropper_layer_image=inner["vlm_frame_cropper_layer_image"],
                                vlm_frame_cropper_layer_bbox=tuple(int(x) for x in inner["vlm_frame_cropper_layer_bbox"]),
                            )
                        )
                        dispatches.append(dispatch)
                        if len(dispatch_pkgs) >= vlm_batch_size:
                            break

                    if dispatch_pkgs:
                        raw_results = run_vlm_inference_batch(
                            vlm_state,
                            dispatch_pkgs,
                            [vlm_query_type] * len(dispatch_pkgs),
                        )
                        vlm_calls += len(raw_results)
                        for raw_result, dispatch in zip(raw_results, dispatches, strict=True):
                            ack = build_vlm_ack_package_from_result(raw_result)
                            register_vlm_ack_package(crop_cache, ack)
                            rec = {
                                "schema": "pipeline_output_v1",
                                "logged_at_unix_s": time.time(),
                                "input_frame_id": frame_id,
                                "vlm_runtime_mode": vlm_effective_runtime,
                                "dispatch": _dispatch_subset(dispatch),
                                "vlm": build_vlm_output_json(raw_result, include_raw_result=bool(args.include_raw_vlm)),
                            }
                            _append_jsonl(out_f, rec)
                            lines_written += 1

            if scene_state is not None and run_scene_awareness_inference is not None:
                _ = run_scene_awareness_inference(scene_state, input_pkg)

            frames_done += 1

        input_layer.close_input_layer()

        if vlm_worker is not None:
            try:
                print("[pipeline] Draining VLM worker...")
                deadline = time.perf_counter() + 30.0
                while time.perf_counter() < deadline:
                    drain_vlm_worker(out_f, last_frame_id)
                    st = vlm_worker.get_status()
                    if st["queue_size"] == 0 and not st["busy"]:
                        break
                    time.sleep(0.02)
                drain_vlm_worker(out_f, last_frame_id)
            except Exception as exc:
                print(f"[pipeline] VLM drain error (non-fatal): {exc}")
            try:
                vlm_worker.shutdown()
            except Exception:
                pass  # suppress thread cleanup crash

    print(f"[pipeline] frames={frames_done} vlm_completions_logged={lines_written} output={out_path}")
    print(f"[pipeline] vlm_runtime_mode_effective={vlm_effective_runtime}")


if __name__ == "__main__":
    main()
