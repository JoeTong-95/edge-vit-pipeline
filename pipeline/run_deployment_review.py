#!/usr/bin/env python3
"""
Run a headless pipeline pass and generate review-package artifacts.

Outputs:
  review-package/runs/<run_id>/new_tracks/
  review-package/runs/<run_id>/vlm_accepted_targets/
  review-package/runs/<run_id>/metadata/new_tracks.csv
  review-package/runs/<run_id>/metadata/vlm_accepted_targets.csv
  review-package/runs/<run_id>/metadata/run_events.jsonl
  review-package/runs/<run_id>/summaries/run_summary.json
  review-package/runs/<run_id>/summaries/run_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

CONFIG_DIR = SRC_DIR / "configuration-layer"
INPUT_DIR = SRC_DIR / "input-layer"
ROI_DIR = SRC_DIR / "roi-layer"
YOLO_DIR = SRC_DIR / "yolo-layer"
TRACKING_DIR = SRC_DIR / "tracking-layer"
VSTATE_DIR = SRC_DIR / "vehicle-state-layer"
VLM_DIR = SRC_DIR / "vlm-layer"
CROPPER_DIR = SRC_DIR / "vlm-frame-cropper-layer"

for p in (
    CONFIG_DIR,
    INPUT_DIR,
    ROI_DIR,
    YOLO_DIR,
    TRACKING_DIR,
    VSTATE_DIR,
    VLM_DIR,
    CROPPER_DIR,
):
    if p.exists():
        sys.path.insert(0, str(p))


TARGET_CLASSES = {"pickup", "van", "truck", "bus"}


NEW_TRACK_COLUMNS = [
    "run_id",
    "source_video",
    "frame_index",
    "track_id",
    "tracker_status",
    "target_class",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "tracker_confidence",
    "image_relpath",
    "review_status",
]

VLM_ACCEPTED_COLUMNS = [
    "run_id",
    "source_video",
    "frame_index",
    "track_id",
    "target_class",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "tracker_confidence",
    "dispatch_mode",
    "dispatch_reason",
    "dispatch_cached_crop_count",
    "from_cache",
    "model_id",
    "query_type",
    "is_type",
    "ack_status",
    "retry_reasons",
    "image_relpath",
    "raw_event_relpath",
]


def _resolve_repo_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sanitize_token(value: str, fallback: str = "unknown") -> str:
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in (value or "").strip())
    safe = safe.strip("._")
    return safe or fallback


def _normalize_bbox(frame_shape: tuple[int, ...], bbox: tuple[Any, Any, Any, Any]) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1 = max(0, min(int(float(bbox[0])), width - 1))
    y1 = max(0, min(int(float(bbox[1])), height - 1))
    x2 = max(x1 + 1, min(int(float(bbox[2])), width))
    y2 = max(y1 + 1, min(int(float(bbox[3])), height))
    return (x1, y1, x2, y2)


def _crop_from_frame(frame_bgr: Any, bbox: tuple[Any, Any, Any, Any]) -> tuple[Any, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = _normalize_bbox(frame_bgr.shape, bbox)
    return frame_bgr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def _append_jsonl(fh: Any, record: dict[str, Any]) -> None:
    fh.write(json.dumps(record, default=str) + "\n")
    fh.flush()


def _to_relpath(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve())).replace("\\", "/")


def _build_run_id(source_video: str, config_tag: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    video_stem = _sanitize_token(Path(source_video).stem or "source")
    tag = _sanitize_token(config_tag or "default")
    return f"{ts}_{video_stem}_{tag}"


def _write_summary_markdown(summary_path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Run Summary",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Source video: `{summary['source_video']}`",
        f"- Frames processed: `{summary['frames_processed']}`",
        f"- New track images saved: `{summary['new_tracks_saved']}`",
        f"- VLM dispatches saved: `{summary['vlm_dispatches_saved']}`",
        f"- VLM results logged: `{summary['vlm_results_logged']}`",
        f"- Duration sec: `{summary['duration_sec']}`",
        f"- VLM enabled: `{summary['vlm_enabled']}`",
        f"- VLM available: `{summary['vlm_available']}`",
        f"- VLM backend: `{summary['vlm_backend']}`",
        f"- VLM model: `{summary['vlm_model']}`",
        f"- VLM runtime mode effective: `{summary['vlm_runtime_mode_effective']}`",
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def _serialize_config_snapshot(cfg: Any) -> dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    return {"raw_repr": repr(cfg)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate review-package artifacts from a headless pipeline run.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 means until source ends).")
    parser.add_argument(
        "--config",
        default="src/configuration-layer/config.yaml",
        help="Repo-relative or absolute path to the configuration YAML to load.",
    )
    parser.add_argument("--video", default="", help="Optional source video override.")
    parser.add_argument("--run-id", default="", help="Optional run id override.")
    parser.add_argument("--config-tag", default="", help="Optional tag for run id.")
    parser.add_argument("--review-root", default="review-package", help="Review package root path.")
    parser.add_argument("--include-raw-vlm", action="store_true", help="Include raw VLM output in per-dispatch event JSON.")
    parser.add_argument("--device", default="", help="Optional override for config_device.")
    parser.add_argument("--yolo-model", default="", help="Optional override for config_yolo_model.")
    parser.add_argument("--disable-vlm", action="store_true", help="Force-disable VLM for this run.")
    args = parser.parse_args()

    from config_node import get_config_value, load_config, validate_config
    from detector import (
        build_yolo_layer_package,
        filter_yolo_detections,
        initialize_yolo_layer,
        run_yolo_detection,
    )
    from input_layer import InputLayer
    from layer import (
        VLMConfig,
        VLMFrameCropperLayerPackage,
        build_vlm_ack_package_from_result,
        build_vlm_output_json,
        initialize_vlm_layer,
        prepare_vlm_prompt,
        run_vlm_inference_batch,
    )
    from roi_layer import build_roi_layer_package, initialize_roi_layer, update_roi_state
    from tracker import assign_tracking_status, build_tracking_layer_package, initialize_tracking_layer, update_tracks
    from vehicle_state_layer import initialize_vehicle_state_layer, update_vehicle_state_from_tracking
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

    config_path = Path(_resolve_repo_path(args.config))
    if not config_path.exists():
        raise FileNotFoundError(f"Config path not found: {config_path}")
    cfg = load_config(config_path)
    validate_config(cfg)

    input_source = str(get_config_value(cfg, "config_input_source"))
    if input_source not in ("video", "camera"):
        raise RuntimeError("run_deployment_review.py supports only config_input_source: video|camera.")

    configured_video_path = str(get_config_value(cfg, "config_input_path"))
    source_video = args.video.strip() or configured_video_path
    resolved_video = _resolve_repo_path(source_video)
    if input_source == "video" and not os.path.exists(resolved_video):
        raise FileNotFoundError(f"Video path not found: {resolved_video}")

    frame_resolution = tuple(get_config_value(cfg, "config_frame_resolution"))
    device = args.device.strip() or str(get_config_value(cfg, "config_device"))
    model = args.yolo_model.strip() or str(get_config_value(cfg, "config_yolo_model"))
    conf = float(get_config_value(cfg, "config_yolo_confidence_threshold"))
    roi_enabled = bool(get_config_value(cfg, "config_roi_enabled"))
    roi_threshold = int(get_config_value(cfg, "config_roi_vehicle_count_threshold"))

    vlm_enabled = bool(get_config_value(cfg, "config_vlm_enabled")) and not bool(args.disable_vlm)
    vlm_backend = str(get_config_value(cfg, "config_vlm_backend"))
    vlm_model = str(get_config_value(cfg, "config_vlm_model"))
    vlm_api_key_env = str(get_config_value(cfg, "config_vlm_api_key_env"))
    vlm_crop_cache_size = int(get_config_value(cfg, "config_vlm_crop_cache_size"))
    vlm_dead_after_lost_frames = int(get_config_value(cfg, "config_vlm_dead_after_lost_frames"))
    _vlm_dev = str(get_config_value(cfg, "config_vlm_device") or "").strip()
    vlm_infer_device = _vlm_dev if _vlm_dev else device

    default_config_tag = args.config_tag.strip()
    if not default_config_tag:
        default_config_tag = f"{Path(model).stem}_{vlm_backend}_{vlm_infer_device}"
    run_id = args.run_id.strip() or _build_run_id(source_video=source_video, config_tag=default_config_tag)

    review_root = Path(_resolve_repo_path(args.review_root))
    run_dir = review_root / "runs" / run_id
    new_tracks_dir = run_dir / "new_tracks"
    accepted_dir = run_dir / "vlm_accepted_targets"
    metadata_dir = run_dir / "metadata"
    summaries_dir = run_dir / "summaries"
    artifacts_dir = run_dir / "artifacts"
    raw_events_dir = artifacts_dir / "vlm_dispatch_events"

    if run_dir.exists():
        shutil.rmtree(run_dir)
    for path in (new_tracks_dir, accepted_dir, metadata_dir, summaries_dir, artifacts_dir, raw_events_dir):
        path.mkdir(parents=True, exist_ok=True)

    (artifacts_dir / "config_snapshot.json").write_text(
        json.dumps(_serialize_config_snapshot(cfg), indent=2, default=str),
        encoding="utf-8",
    )

    new_tracks_csv_path = metadata_dir / "new_tracks.csv"
    accepted_csv_path = metadata_dir / "vlm_accepted_targets.csv"
    events_jsonl_path = metadata_dir / "run_events.jsonl"

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
            config_input_path=resolved_video,
        )

    initialize_roi_layer(config_roi_enabled=roi_enabled, config_roi_vehicle_count_threshold=roi_threshold)
    initialize_yolo_layer(model_name=model, conf_threshold=conf, device=device)
    initialize_tracking_layer(frame_rate=30)
    initialize_vehicle_state_layer(prune_after_lost_frames=None)

    vlm_available = False
    vlm_state = None
    crop_cache = None
    vlm_query_type = "vehicle_semantics_single_shot_v1"
    if vlm_enabled:
        try:
            vlm_state = initialize_vlm_layer(
                VLMConfig(
                    config_vlm_enabled=True,
                    config_vlm_backend=vlm_backend,
                    config_vlm_model=_resolve_repo_path(vlm_model),
                    config_vlm_api_key_env=vlm_api_key_env,
                    config_device=vlm_infer_device,
                )
            )
            crop_cache = initialize_vlm_crop_cache(
                config_vlm_crop_cache_size=vlm_crop_cache_size,
                config_vlm_dead_after_lost_frames=vlm_dead_after_lost_frames,
            )
            vlm_available = True
        except Exception as exc:
            print(f"[review-run] VLM unavailable ({type(exc).__name__}: {exc}); continuing without VLM.")
            vlm_available = False

    saved_new_track_ids: set[str] = set()
    frames_done = 0
    new_tracks_saved = 0
    vlm_dispatches_saved = 0
    vlm_results_logged = 0
    started_at = time.time()

    with (
        new_tracks_csv_path.open("w", newline="", encoding="utf-8") as new_tracks_f,
        accepted_csv_path.open("w", newline="", encoding="utf-8") as accepted_f,
        events_jsonl_path.open("w", encoding="utf-8") as events_f,
    ):
        new_tracks_writer = csv.DictWriter(new_tracks_f, fieldnames=NEW_TRACK_COLUMNS)
        accepted_writer = csv.DictWriter(accepted_f, fieldnames=VLM_ACCEPTED_COLUMNS)
        new_tracks_writer.writeheader()
        accepted_writer.writeheader()

        max_frames = int(args.max_frames)
        while True:
            if max_frames > 0 and frames_done >= max_frames:
                break

            raw = input_layer.read_next_frame()
            if raw is None:
                if input_source == "camera":
                    time.sleep(0.01)
                    continue
                break

            pkg = input_layer.build_input_layer_package(raw)
            input_pkg = {
                "input_layer_frame_id": pkg.input_layer_frame_id,
                "input_layer_timestamp": pkg.input_layer_timestamp,
                "input_layer_image": pkg.input_layer_image,
                "input_layer_source_type": pkg.input_layer_source_type,
                "input_layer_resolution": pkg.input_layer_resolution,
            }
            frame_id = int(input_pkg["input_layer_frame_id"])

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

            track_ids = tracking_pkg.get("tracking_layer_track_id", [])
            bboxes = tracking_pkg.get("tracking_layer_bbox", [])
            classes = tracking_pkg.get("tracking_layer_detector_class", [])
            confidences = tracking_pkg.get("tracking_layer_confidence", [])
            statuses = tracking_pkg.get("tracking_layer_status", [])

            for i in range(len(track_ids)):
                track_id = str(track_ids[i])
                target_class = str(classes[i])
                confidence = float(confidences[i])
                status = str(statuses[i])
                bbox = tuple(bboxes[i])
                frame_bgr = input_pkg["input_layer_image"]

                if status == "new" and target_class in TARGET_CLASSES and track_id not in saved_new_track_ids:
                    crop_img, clamped_bbox = _crop_from_frame(frame_bgr, bbox)
                    if crop_img.size > 0:
                        filename = (
                            f"track_{_sanitize_token(track_id)}"
                            f"__frame_{frame_id}"
                            f"__class_{_sanitize_token(target_class)}.jpg"
                        )
                        image_path = new_tracks_dir / filename
                        cv2.imwrite(str(image_path), crop_img)
                        image_relpath = _to_relpath(image_path, review_root)
                        row = {
                            "run_id": run_id,
                            "source_video": source_video,
                            "frame_index": frame_id,
                            "track_id": track_id,
                            "tracker_status": status,
                            "target_class": target_class,
                            "bbox_x1": clamped_bbox[0],
                            "bbox_y1": clamped_bbox[1],
                            "bbox_x2": clamped_bbox[2],
                            "bbox_y2": clamped_bbox[3],
                            "tracker_confidence": confidence,
                            "image_relpath": image_relpath,
                            "review_status": "",
                        }
                        new_tracks_writer.writerow(row)
                        saved_new_track_ids.add(track_id)
                        new_tracks_saved += 1
                        _append_jsonl(
                            events_f,
                            {
                                "event_type": "new_track_saved",
                                "timestamp_utc": _utc_now_iso(),
                                "run_id": run_id,
                                "source_video": source_video,
                                "frame_index": frame_id,
                                "track_id": track_id,
                                "payload": row,
                            },
                        )

                if not (vlm_enabled and vlm_available and crop_cache is not None):
                    continue
                row = {
                    "track_id": track_id,
                    "bbox": bbox,
                    "detector_class": target_class,
                    "confidence": confidence,
                    "status": status,
                }
                refresh_vlm_crop_cache_track_state(crop_cache, row, frame_id)
                if status == "lost":
                    continue
                req = build_vlm_frame_cropper_request_package(
                    input_layer_package=input_pkg,
                    tracking_layer_package=tracking_pkg,
                    track_index=i,
                    vlm_frame_cropper_trigger_reason=f"review_tracking_status:{status}",
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
                    frame_id,
                    f"review_tracking_status:{status}",
                )

            if vlm_enabled and vlm_available and crop_cache is not None and vlm_state is not None:
                track_cache_keys = sorted(crop_cache["track_caches"].keys(), key=lambda x: (len(str(x)), str(x)))
                for tid in track_cache_keys:
                    dispatch = build_vlm_dispatch_package(crop_cache, tid)
                    if dispatch is None:
                        continue

                    inner = dispatch["vlm_frame_cropper_layer_package"]
                    track_cache = crop_cache["track_caches"].get(str(tid), {})
                    sent_pkg = track_cache.get("vlm_last_sent_package") or {}
                    dispatch_bbox = tuple(inner["vlm_frame_cropper_layer_bbox"])
                    bbox = tuple(sent_pkg.get("bbox") or dispatch_bbox)
                    confidence = float(sent_pkg.get("confidence") or 0.0)
                    target_class = str(track_cache.get("detector_class") or "")
                    dispatch_frame_id = int(sent_pkg.get("frame_id") or frame_id)

                    crop_img = inner["vlm_frame_cropper_layer_image"]
                    image_name = (
                        f"track_{_sanitize_token(str(inner['vlm_frame_cropper_layer_track_id']))}"
                        f"__frame_{dispatch_frame_id}"
                        f"__dispatch_{_sanitize_token(str(dispatch.get('vlm_dispatch_mode') or 'unknown'))}"
                        f"__class_{_sanitize_token(target_class or 'unknown')}.jpg"
                    )
                    image_path = accepted_dir / image_name
                    cv2.imwrite(str(image_path), crop_img)
                    image_relpath = _to_relpath(image_path, review_root)

                    vlm_crop_pkg = VLMFrameCropperLayerPackage(
                        vlm_frame_cropper_layer_track_id=str(inner["vlm_frame_cropper_layer_track_id"]),
                        vlm_frame_cropper_layer_image=inner["vlm_frame_cropper_layer_image"],
                        vlm_frame_cropper_layer_bbox=tuple(int(x) for x in inner["vlm_frame_cropper_layer_bbox"]),
                    )
                    prompt_text = prepare_vlm_prompt(vlm_query_type, vlm_crop_pkg)
                    raw_result = run_vlm_inference_batch(vlm_state, [vlm_crop_pkg], [vlm_query_type])[0]
                    ack = build_vlm_ack_package_from_result(raw_result)
                    register_vlm_ack_package(crop_cache, ack)
                    vlm_output = build_vlm_output_json(raw_result, include_raw_result=bool(args.include_raw_vlm))
                    normalized = vlm_output.get("normalized_result", {})

                    raw_event = {
                        "schema": "review_dispatch_event_v1",
                        "run_id": run_id,
                        "source_video": source_video,
                        "frame_index": dispatch_frame_id,
                        "track_id": str(inner["vlm_frame_cropper_layer_track_id"]),
                        "dispatch": {
                            "vlm_dispatch_track_id": dispatch.get("vlm_dispatch_track_id"),
                            "vlm_dispatch_mode": dispatch.get("vlm_dispatch_mode"),
                            "vlm_dispatch_reason": dispatch.get("vlm_dispatch_reason"),
                            "vlm_dispatch_cached_crop_count": dispatch.get("vlm_dispatch_cached_crop_count"),
                        },
                        "prompt_text": prompt_text,
                        "vlm": vlm_output,
                        "timestamp_utc": _utc_now_iso(),
                    }
                    raw_event_path = raw_events_dir / f"dispatch_{vlm_dispatches_saved + 1:06d}.json"
                    raw_event_path.write_text(json.dumps(raw_event, indent=2), encoding="utf-8")
                    raw_event_relpath = _to_relpath(raw_event_path, review_root)

                    accepted_row = {
                        "run_id": run_id,
                        "source_video": source_video,
                        "frame_index": dispatch_frame_id,
                        "track_id": str(inner["vlm_frame_cropper_layer_track_id"]),
                        "target_class": target_class,
                        "bbox_x1": int(bbox[0]),
                        "bbox_y1": int(bbox[1]),
                        "bbox_x2": int(bbox[2]),
                        "bbox_y2": int(bbox[3]),
                        "tracker_confidence": confidence,
                        "dispatch_mode": str(dispatch.get("vlm_dispatch_mode") or ""),
                        "dispatch_reason": str(dispatch.get("vlm_dispatch_reason") or ""),
                        "dispatch_cached_crop_count": int(dispatch.get("vlm_dispatch_cached_crop_count") or 0),
                        "from_cache": True,
                        "model_id": str(vlm_output.get("vlm_layer_model_id") or ""),
                        "query_type": str(vlm_output.get("vlm_layer_query_type") or ""),
                        "is_type": bool(normalized.get("is_truck")),
                        "ack_status": str(normalized.get("vlm_ack_status") or ""),
                        "retry_reasons": json.dumps(list(normalized.get("vlm_retry_reasons") or [])),
                        "image_relpath": image_relpath,
                        "raw_event_relpath": raw_event_relpath,
                    }
                    accepted_writer.writerow(accepted_row)
                    vlm_dispatches_saved += 1
                    vlm_results_logged += 1

                    _append_jsonl(
                        events_f,
                        {
                            "event_type": "vlm_dispatch_saved",
                            "timestamp_utc": _utc_now_iso(),
                            "run_id": run_id,
                            "source_video": source_video,
                            "frame_index": dispatch_frame_id,
                            "track_id": str(inner["vlm_frame_cropper_layer_track_id"]),
                            "payload": {
                                "image_relpath": image_relpath,
                                "raw_event_relpath": raw_event_relpath,
                                "dispatch_mode": dispatch.get("vlm_dispatch_mode"),
                                "dispatch_reason": dispatch.get("vlm_dispatch_reason"),
                                "dispatch_cached_crop_count": dispatch.get("vlm_dispatch_cached_crop_count"),
                            },
                        },
                    )
                    _append_jsonl(
                        events_f,
                        {
                            "event_type": "vlm_result_logged",
                            "timestamp_utc": _utc_now_iso(),
                            "run_id": run_id,
                            "source_video": source_video,
                            "frame_index": dispatch_frame_id,
                            "track_id": str(inner["vlm_frame_cropper_layer_track_id"]),
                            "payload": {
                                "is_type": bool(normalized.get("is_truck")),
                                "ack_status": str(normalized.get("vlm_ack_status") or ""),
                                "retry_reasons": list(normalized.get("vlm_retry_reasons") or []),
                            },
                        },
                    )

            frames_done += 1

    input_layer.close_input_layer()

    summary = {
        "schema": "review_run_summary_v1",
        "run_id": run_id,
        "source_video": source_video,
        "frames_processed": frames_done,
        "new_tracks_saved": new_tracks_saved,
        "vlm_dispatches_saved": vlm_dispatches_saved,
        "vlm_results_logged": vlm_results_logged,
        "duration_sec": round(time.time() - started_at, 3),
        "vlm_enabled": bool(vlm_enabled),
        "vlm_available": bool(vlm_available),
        "vlm_backend": vlm_backend,
        "vlm_model": vlm_model,
        "vlm_runtime_mode_effective": "inline",
        "created_at_utc": _utc_now_iso(),
    }
    summary_json_path = summaries_dir / "run_summary.json"
    summary_md_path = summaries_dir / "run_summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_markdown(summary_md_path, summary)

    with events_jsonl_path.open("a", encoding="utf-8") as events_f:
        _append_jsonl(
            events_f,
            {
                "event_type": "run_summary",
                "timestamp_utc": _utc_now_iso(),
                "run_id": run_id,
                "source_video": source_video,
                "frame_index": -1,
                "payload": summary,
            },
        )

    print(f"[review-run] run_id={run_id}")
    print(f"[review-run] frames={frames_done} new_tracks={new_tracks_saved} vlm_dispatches={vlm_dispatches_saved}")
    print(f"[review-run] output={run_dir}")


if __name__ == "__main__":
    main()
