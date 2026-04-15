#!/usr/bin/env python3
"""
benchmark.py

Runs a short, end-to-end *video-only* pipeline pass and prints timing + throughput
to help assess device usability for this stack.

It reads `src/configuration-layer/config.yaml` and exercises:
input(video) -> ROI (if enabled) -> YOLO -> tracking -> vehicle-state
optionally scene-awareness (if enabled), and optionally VLM inference (if enabled).

When `config_vlm_enabled` is true and PyTorch + transformers are available, this
script will also run the cropper->dispatch->VLM inference loop (synchronously)
and report VLM latency and VLM calls per processed frame.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

# ---------------------------------------------------------------------------
# Benchmark controls (edit these; no CLI flags needed)
# ---------------------------------------------------------------------------
# For steady-state benchmarking, keep MEASURE_SECONDS at 60 and WARMUP_SECONDS
# non-zero so cold-start effects are excluded from the reported averages.
WARMUP_SECONDS = 10.0
MEASURE_SECONDS = 60.0

# If True, the benchmark will only start measurement AFTER ROI has locked.
# This avoids misleading "ROI speedup" numbers computed from tiny post-lock windows.
# This does NOT change runtime ROI behavior; it only changes how benchmarking windows are selected.
MEASURE_ONLY_AFTER_ROI_LOCK = True

# Optional override for benchmarking comparisons without editing config.yaml:
# - set to True/False to force ROI on/off in this benchmark process
# - set to None to use config.yaml
BENCH_OVERRIDE_ROI_ENABLED: bool | None = None

# Optional override for benchmarking different videos without editing config.yaml.
# When set, this should be a repo-relative path like "data/sample1.mp4" or an absolute path.
BENCH_OVERRIDE_INPUT_PATH: str | None = None

# Fallback (frame-count mode). Used only if MEASURE_SECONDS <= 0.
WARMUP_FRAMES = 3
MEASURED_FRAMES = 30

CONFIG_DIR = SRC_DIR / "configuration-layer"
INPUT_DIR = SRC_DIR / "input-layer"
ROI_DIR = SRC_DIR / "roi-layer"
YOLO_DIR = SRC_DIR / "yolo-layer"
TRACKING_DIR = SRC_DIR / "tracking-layer"
VSTATE_DIR = SRC_DIR / "vehicle-state-layer"
SCENE_DIR = SRC_DIR / "scene-awareness-layer"
VLM_DIR = SRC_DIR / "vlm-layer"
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
    CROPPER_DIR,
):
    if p.exists():
        sys.path.insert(0, str(p))


def _resolve_repo_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _p(label: str, value: str) -> None:
    print(f"[profile] {label}: {value}")


def _safe_div(n: float, d: float) -> float:
    return n / d if d not in (0.0, -0.0) else 0.0


def _fmt_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def main() -> None:
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401

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
    if input_source != "video":
        raise RuntimeError("benchmark.py is video-only. Set config_input_source=video.")

    configured_video_path = str(get_config_value(cfg, "config_input_path"))
    video_path_value = BENCH_OVERRIDE_INPUT_PATH if BENCH_OVERRIDE_INPUT_PATH else configured_video_path
    video_path = _resolve_repo_path(str(video_path_value))
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Configured video path not found: {video_path}")

    frame_resolution = tuple(get_config_value(cfg, "config_frame_resolution"))
    device = str(get_config_value(cfg, "config_device"))
    model = str(get_config_value(cfg, "config_yolo_model"))
    conf = float(get_config_value(cfg, "config_yolo_confidence_threshold"))
    roi_enabled = bool(get_config_value(cfg, "config_roi_enabled"))
    if BENCH_OVERRIDE_ROI_ENABLED is not None:
        roi_enabled = bool(BENCH_OVERRIDE_ROI_ENABLED)
    roi_threshold = int(get_config_value(cfg, "config_roi_vehicle_count_threshold"))
    scene_enabled = bool(get_config_value(cfg, "config_scene_awareness_enabled"))

    vlm_enabled = bool(get_config_value(cfg, "config_vlm_enabled"))
    vlm_model = str(get_config_value(cfg, "config_vlm_model"))
    vlm_crop_cache_size = int(get_config_value(cfg, "config_vlm_crop_cache_size"))
    vlm_dead_after_lost_frames = int(get_config_value(cfg, "config_vlm_dead_after_lost_frames"))

    # Profile settings
    warmup_frames = int(WARMUP_FRAMES)
    measured_frames = int(MEASURED_FRAMES)
    warmup_seconds = float(WARMUP_SECONDS or 0.0)
    measure_seconds = float(MEASURE_SECONDS or 0.0)
    use_time_budget = (warmup_seconds > 0.0) or (measure_seconds > 0.0)
    if use_time_budget:
        if warmup_seconds < 0.0 or measure_seconds <= 0.0:
            raise ValueError("For time-based mode, require MEASURE_SECONDS > 0 and WARMUP_SECONDS >= 0.")
    else:
        if warmup_frames < 0 or measured_frames <= 0:
            raise ValueError("For frame-based mode, require MEASURED_FRAMES > 0 and WARMUP_FRAMES >= 0.")

    _p("branch", _git_branch())
    _p("video", video_path)
    if BENCH_OVERRIDE_INPUT_PATH:
        _p("video_override", str(BENCH_OVERRIDE_INPUT_PATH))
    try:
        cap = cv2.VideoCapture(video_path)
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
    except Exception:
        src_fps = 0.0
        src_frames = 0
    _p("source_fps", f"{src_fps:.2f}" if src_fps > 0 else "unknown")
    _p("source_total_frames", str(src_frames) if src_frames > 0 else "unknown")
    _p("frame_resolution", f"{frame_resolution[0]}x{frame_resolution[1]}")
    _p("device", device)
    _p("yolo_model", model)
    _p("yolo_conf", str(conf))
    _p("roi_enabled", str(roi_enabled))
    _p("measure_only_after_roi_lock", str(bool(MEASURE_ONLY_AFTER_ROI_LOCK and roi_enabled)))
    _p("scene_enabled", str(scene_enabled and scene_available))
    _p("vlm_enabled", str(vlm_enabled))
    _p("vlm_model", _resolve_repo_path(vlm_model) if vlm_model else "none")
    if use_time_budget:
        _p("warmup_seconds", f"{warmup_seconds:.1f}")
        _p("measure_seconds", f"{measure_seconds:.1f}")
    else:
        _p("warmup_frames", str(warmup_frames))
        _p("measured_frames", str(measured_frames))

    input_layer = InputLayer()
    input_layer.initialize_input_layer(
        config_input_source="video",
        config_frame_resolution=frame_resolution,
        config_input_path=video_path,
    )

    initialize_roi_layer(config_roi_enabled=roi_enabled, config_roi_vehicle_count_threshold=roi_threshold)
    initialize_yolo_layer(model_name=model, conf_threshold=conf, device=device)
    initialize_tracking_layer(frame_rate=30)
    initialize_vehicle_state_layer(prune_after_lost_frames=None)

    # Initialize crop cache + VLM runtime if enabled and deps exist.
    vlm_available = False
    vlm_state = None
    crop_cache = None
    vlm_calls = 0
    vlm_query_type = "vehicle_semantics_single_shot_v1"
    vlm_query_s_total = 0.0
    vlm_query_count = 0
    vlm_runtime_device = "disabled"

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
                VLMFrameCropperLayerPackage,
                build_vlm_ack_package_from_result,
                initialize_vlm_layer,
                run_vlm_inference,
            )

            vlm_state = initialize_vlm_layer(
                VLMConfig(
                    config_vlm_enabled=True,
                    config_vlm_model=_resolve_repo_path(vlm_model),
                    config_device=device,
                )
            )
            vlm_runtime_device = str(getattr(vlm_state, "vlm_runtime_device", "unknown"))
            crop_cache = initialize_vlm_crop_cache(
                config_vlm_crop_cache_size=vlm_crop_cache_size,
                config_vlm_dead_after_lost_frames=vlm_dead_after_lost_frames,
            )
            vlm_available = True
        except Exception as exc:
            _p("vlm_status", f"SKIP ({type(exc).__name__}: {exc})")
            vlm_available = False

    scene_state = None
    if scene_enabled and scene_available and initialize_scene_awareness_layer is not None:
        scene_state = initialize_scene_awareness_layer(config_scene_awareness_enabled=True, config_device=device)

    totals = {
        "input_s": 0.0,
        "roi_s": 0.0,
        "yolo_s": 0.0,
        "tracking_s": 0.0,
        "state_s": 0.0,
        "vlm_s": 0.0,
        "scene_s": 0.0,
        "end_to_end_s": 0.0,
    }
    det_counts = 0
    track_counts = 0
    roi_lock_frame_id: int | None = None
    roi_lock_candidate_count: int | None = None
    roi_lock_bounds: tuple[int, int, int, int] | None = None
    roi_last_candidate_count = 0
    roi_last_locked = False
    roi_last_bounds: tuple[int, int, int, int] | None = None

    yolo_full_s_total = 0.0
    yolo_full_frames = 0
    yolo_roi_s_total = 0.0
    yolo_roi_frames = 0
    yolo_full_infer_s_total = 0.0
    yolo_full_post_s_total = 0.0
    yolo_roi_infer_s_total = 0.0
    yolo_roi_post_s_total = 0.0
    yolo_full_pixels_total = 0
    yolo_roi_pixels_total = 0
    yolo_full_det_total = 0
    yolo_roi_det_total = 0

    def process_one() -> bool:
        nonlocal det_counts, track_counts, vlm_calls
        nonlocal roi_lock_frame_id, roi_lock_candidate_count, roi_lock_bounds
        nonlocal roi_last_candidate_count, roi_last_locked, roi_last_bounds
        nonlocal yolo_full_s_total, yolo_full_frames, yolo_roi_s_total, yolo_roi_frames
        nonlocal yolo_full_infer_s_total, yolo_full_post_s_total, yolo_roi_infer_s_total, yolo_roi_post_s_total
        nonlocal yolo_full_pixels_total, yolo_roi_pixels_total, yolo_full_det_total, yolo_roi_det_total
        nonlocal vlm_query_s_total, vlm_query_count
        t0 = time.perf_counter()

        t_in0 = time.perf_counter()
        raw = input_layer.read_next_frame()
        if raw is None:
            return False
        pkg = input_layer.build_input_layer_package(raw)
        input_pkg = {
            "input_layer_frame_id": pkg.input_layer_frame_id,
            "input_layer_timestamp": pkg.input_layer_timestamp,
            "input_layer_image": pkg.input_layer_image,
            "input_layer_source_type": pkg.input_layer_source_type,
            "input_layer_resolution": pkg.input_layer_resolution,
        }
        t_in1 = time.perf_counter()

        t_roi0 = time.perf_counter()
        roi_pkg = build_roi_layer_package(input_pkg)
        t_roi1 = time.perf_counter()

        dets_boot = run_yolo_detection(roi_pkg if roi_pkg.get("roi_layer_locked") else input_pkg)
        dets_boot = filter_yolo_detections(dets_boot)
        roi_state = update_roi_state(input_pkg, dets_boot)
        roi_last_candidate_count = int(roi_state.get("roi_candidate_box_count", 0))
        roi_last_locked = bool(roi_state.get("roi_layer_locked", False))
        bounds = roi_state.get("roi_layer_bounds")
        roi_last_bounds = tuple(int(v) for v in bounds) if bounds is not None else None
        roi_pkg = build_roi_layer_package(input_pkg)
        if roi_lock_frame_id is None and roi_last_locked:
            roi_lock_frame_id = int(input_pkg["input_layer_frame_id"])
            roi_lock_candidate_count = roi_last_candidate_count
            if roi_last_bounds is not None:
                roi_lock_bounds = roi_last_bounds

        t_y0 = time.perf_counter()
        yolo_using_roi = bool(roi_enabled and roi_pkg.get("roi_layer_locked"))
        yolo_upstream = roi_pkg if yolo_using_roi else input_pkg
        yolo_img = yolo_upstream.get("roi_layer_image", yolo_upstream.get("input_layer_image"))
        if yolo_img is not None:
            h, w = yolo_img.shape[:2]
            if yolo_using_roi:
                yolo_roi_pixels_total += int(h) * int(w)
            else:
                yolo_full_pixels_total += int(h) * int(w)
        t_inf0 = time.perf_counter()
        dets = run_yolo_detection(yolo_upstream)
        t_inf1 = time.perf_counter()
        dets = filter_yolo_detections(dets)
        yolo_pkg = build_yolo_layer_package(input_pkg["input_layer_frame_id"], dets)
        t_y1 = time.perf_counter()
        yolo_dt = t_y1 - t_y0
        yolo_infer_dt = t_inf1 - t_inf0
        yolo_post_dt = (t_y1 - t_inf1)
        if yolo_using_roi:
            yolo_roi_s_total += yolo_dt
            yolo_roi_frames += 1
            yolo_roi_infer_s_total += yolo_infer_dt
            yolo_roi_post_s_total += yolo_post_dt
            yolo_roi_det_total += len(dets)
        else:
            yolo_full_s_total += yolo_dt
            yolo_full_frames += 1
            yolo_full_infer_s_total += yolo_infer_dt
            yolo_full_post_s_total += yolo_post_dt
            yolo_full_det_total += len(dets)
        det_counts += len(yolo_pkg.get("yolo_layer_detections", []))

        t_tr0 = time.perf_counter()
        current_tracks = update_tracks(yolo_pkg)
        status_tracks = assign_tracking_status(current_tracks, input_pkg["input_layer_frame_id"])
        tracking_pkg = build_tracking_layer_package(input_pkg["input_layer_frame_id"], status_tracks)
        t_tr1 = time.perf_counter()
        track_counts += len(tracking_pkg.get("tracking_layer_track_id", []))

        t_st0 = time.perf_counter()
        update_vehicle_state_from_tracking(tracking_pkg)
        t_st1 = time.perf_counter()

        t_v0 = time.perf_counter()
        if vlm_enabled and vlm_available and crop_cache is not None and vlm_state is not None:
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
                q0 = time.perf_counter()
                raw_result = run_vlm_inference(vlm_state, vlm_crop_pkg, vlm_query_type)
                q1 = time.perf_counter()
                vlm_query_s_total += q1 - q0
                vlm_query_count += 1
                ack = build_vlm_ack_package_from_result(raw_result)
                register_vlm_ack_package(crop_cache, ack)
                vlm_calls += 1
                break
        t_v1 = time.perf_counter()

        t_sc0 = time.perf_counter()
        if scene_state is not None and run_scene_awareness_inference is not None:
            _ = run_scene_awareness_inference(scene_state, input_pkg)
        t_sc1 = time.perf_counter()

        t1 = time.perf_counter()

        totals["input_s"] += t_in1 - t_in0
        totals["roi_s"] += t_roi1 - t_roi0
        totals["yolo_s"] += t_y1 - t_y0
        totals["tracking_s"] += t_tr1 - t_tr0
        totals["state_s"] += t_st1 - t_st0
        totals["vlm_s"] += t_v1 - t_v0
        totals["scene_s"] += t_sc1 - t_sc0
        totals["end_to_end_s"] += t1 - t0
        return True

    if use_time_budget:
        t_warm_end = time.perf_counter() + warmup_seconds
        while time.perf_counter() < t_warm_end:
            if not process_one():
                break
    else:
        for _ in range(warmup_frames):
            if not process_one():
                break

    # If requested, wait until ROI is locked before starting the measured window.
    # This avoids comparing "full vs ROI" with an undersampled post-lock segment.
    roi_waited_s = 0.0
    if roi_enabled and MEASURE_ONLY_AFTER_ROI_LOCK:
        roi_wait_start = time.perf_counter()
        while roi_lock_frame_id is None:
            if not process_one():
                break
            if (time.perf_counter() - roi_wait_start) > 20.0:
                break
        roi_waited_s = max(0.0, time.perf_counter() - roi_wait_start)

    for k in totals:
        totals[k] = 0.0
    det_counts = 0
    track_counts = 0
    vlm_calls = 0
    vlm_query_s_total = 0.0
    vlm_query_count = 0
    roi_lock_frame_id = None
    roi_lock_candidate_count = None
    roi_lock_bounds = None
    roi_last_candidate_count = 0
    roi_last_locked = False
    roi_last_bounds = None
    yolo_full_s_total = 0.0
    yolo_full_frames = 0
    yolo_roi_s_total = 0.0
    yolo_roi_frames = 0
    yolo_full_infer_s_total = 0.0
    yolo_full_post_s_total = 0.0
    yolo_roi_infer_s_total = 0.0
    yolo_roi_post_s_total = 0.0
    yolo_full_pixels_total = 0
    yolo_roi_pixels_total = 0
    yolo_full_det_total = 0
    yolo_roi_det_total = 0

    frames_done = 0
    measure_started = time.perf_counter()
    if use_time_budget:
        t_measure_end = time.perf_counter() + measure_seconds
        while time.perf_counter() < t_measure_end:
            if not process_one():
                break
            frames_done += 1
    else:
        for _ in range(measured_frames):
            if not process_one():
                break
            frames_done += 1
    measure_elapsed_s = max(1e-9, time.perf_counter() - measure_started)

    input_layer.close_input_layer()

    if frames_done <= 0:
        raise RuntimeError("No frames processed in measured section.")

    _p("frames_processed", str(frames_done))
    _p("avg_detections_per_frame", f"{det_counts / frames_done:.2f}")
    _p("avg_tracks_per_frame", f"{track_counts / frames_done:.2f}")
    _p("avg_vlm_calls_per_frame", f"{vlm_calls / frames_done:.3f}")

    e2e_ms = (totals["end_to_end_s"] / frames_done) * 1000.0
    fps = 1.0 / max(1e-9, totals["end_to_end_s"] / frames_done)
    _p("avg_end_to_end_ms", f"{e2e_ms:.2f}")
    _p("estimated_fps", f"{fps:.2f}")

    _p("avg_input_ms", f"{(totals['input_s'] / frames_done) * 1000.0:.2f}")
    _p("avg_roi_ms", f"{(totals['roi_s'] / frames_done) * 1000.0:.2f}")
    _p("avg_yolo_ms", f"{(totals['yolo_s'] / frames_done) * 1000.0:.2f}")
    _p("avg_tracking_ms", f"{(totals['tracking_s'] / frames_done) * 1000.0:.2f}")
    _p("avg_state_ms", f"{(totals['state_s'] / frames_done) * 1000.0:.2f}")
    _p("avg_vlm_ms", f"{(totals['vlm_s'] / frames_done) * 1000.0:.2f}")
    _p("avg_scene_ms", f"{(totals['scene_s'] / frames_done) * 1000.0:.2f}")
    if roi_enabled and MEASURE_ONLY_AFTER_ROI_LOCK:
        _p("roi_lock_waited_s", f"{roi_waited_s:.2f}")
        _p("roi_locked_before_measure", str(roi_lock_frame_id is not None))

    print()
    avg_yolo_full_s = _safe_div(yolo_full_s_total, float(yolo_full_frames))
    avg_yolo_roi_s = _safe_div(yolo_roi_s_total, float(yolo_roi_frames))
    yolo_full_capacity_fps = _safe_div(1.0, avg_yolo_full_s) if avg_yolo_full_s > 0 else 0.0
    yolo_roi_capacity_fps = _safe_div(1.0, avg_yolo_roi_s) if avg_yolo_roi_s > 0 else 0.0
    avg_yolo_full_infer_s = _safe_div(yolo_full_infer_s_total, float(yolo_full_frames))
    avg_yolo_full_post_s = _safe_div(yolo_full_post_s_total, float(yolo_full_frames))
    avg_yolo_roi_infer_s = _safe_div(yolo_roi_infer_s_total, float(yolo_roi_frames))
    avg_yolo_roi_post_s = _safe_div(yolo_roi_post_s_total, float(yolo_roi_frames))
    avg_vlm_query_ms = (_safe_div(vlm_query_s_total, float(vlm_query_count)) * 1000.0) if vlm_query_count > 0 else 0.0

    print("Layer-by-layer summary (raw + interpretation):")
    print()
    print("Input:")
    if src_fps > 0:
        verdict = "can keep up" if (fps - src_fps) >= -0.5 else "cannot keep up"
        print(f"- source_fps: {src_fps:.2f} | measured_pipeline_fps: {fps:.2f} -> {verdict} vs real-time source")
    else:
        print(f"- source_fps: unknown | measured_pipeline_fps: {fps:.2f}")
    print()

    print("ROI (Layer 3):")
    if roi_enabled:
        if roi_lock_frame_id is None:
            print(
                f"- status: collecting (not locked during measured run) | "
                f"candidates_collected: {roi_last_candidate_count} / threshold {roi_threshold}"
            )
            if roi_last_bounds is not None:
                print(f"- current_bounds: {roi_last_bounds}")
        else:
            print(f"- status: locked at frame {roi_lock_frame_id} after collecting {roi_lock_candidate_count} unique-ish vehicle boxes")
            if roi_lock_bounds is not None:
                x1, y1, x2, y2 = roi_lock_bounds
                roi_area = max(1, (x2 - x1) * (y2 - y1))
                full_area = int(frame_resolution[0]) * int(frame_resolution[1])
                frac = roi_area / max(1, full_area)
                print(f"- bounds: {roi_lock_bounds} | roi_area: {roi_area} px ({_fmt_pct(frac)} of frame)")
                if frac < 0.9:
                    print("- interpretation: ROI is meaningfully smaller than full frame; YOLO should speed up after lock.")
                else:
                    print("- interpretation: ROI is close to full frame; expect little YOLO speedup from ROI.")
    else:
        print("- status: disabled (YOLO always runs on full frame)")
    print()

    print("YOLO (Layer 4):")
    if yolo_full_frames > 0:
        print(f"- pre-ROI frames_measured: {yolo_full_frames}")
        print(f"- pre-ROI avg_yolo_ms: {avg_yolo_full_s * 1000.0:.2f} ({yolo_full_capacity_fps:.2f} fps capacity)")
        print(f"- pre-ROI yolo_infer_ms: {avg_yolo_full_infer_s * 1000.0:.2f} | yolo_post_ms: {avg_yolo_full_post_s * 1000.0:.2f}")
        if src_fps > 0 and yolo_full_capacity_fps > 0:
            print(f"- pre-ROI capacity vs source: {_fmt_pct(yolo_full_capacity_fps / src_fps)} of source fps")
        print(f"- pre-ROI avg_pixels: {int(_safe_div(float(yolo_full_pixels_total), float(yolo_full_frames))):,} px/frame")
        print(f"- pre-ROI avg_detections: {_safe_div(float(yolo_full_det_total), float(yolo_full_frames)):.2f} / frame")
    if yolo_roi_frames > 0:
        print(f"- post-ROI frames_measured: {yolo_roi_frames}")
        print(f"- post-ROI avg_yolo_ms: {avg_yolo_roi_s * 1000.0:.2f} ({yolo_roi_capacity_fps:.2f} fps capacity)")
        print(f"- post-ROI yolo_infer_ms: {avg_yolo_roi_infer_s * 1000.0:.2f} | yolo_post_ms: {avg_yolo_roi_post_s * 1000.0:.2f}")
        if src_fps > 0 and yolo_roi_capacity_fps > 0:
            print(f"- post-ROI capacity vs source: {_fmt_pct(yolo_roi_capacity_fps / src_fps)} of source fps")
        print(f"- post-ROI avg_pixels: {int(_safe_div(float(yolo_roi_pixels_total), float(yolo_roi_frames))):,} px/frame")
        print(f"- post-ROI avg_detections: {_safe_div(float(yolo_roi_det_total), float(yolo_roi_frames)):.2f} / frame")
        if avg_yolo_full_s > 0 and avg_yolo_roi_s > 0:
            print(f"- ROI speedup factor: {avg_yolo_full_s / avg_yolo_roi_s:.2f}x")
    print("- interpretation: if YOLO capacity is < source fps, YOLO is a real-time bottleneck at this resolution/model.")
    print()

    print("VLM (Layer 8):")
    if not vlm_enabled:
        print("- status: disabled")
    elif not vlm_available:
        print("- status: enabled in config but skipped (missing deps / model load failure)")
    else:
        print(f"- runtime_device: {vlm_runtime_device} (requested config_device={device})")
        if vlm_query_count > 0:
            print(f"- vlm_query_time: {avg_vlm_query_ms:.1f} ms/query (avg over {vlm_query_count} queries)")
        else:
            print("- vlm_query_time: no queries executed (no dispatches during run)")
        print(f"- vlm_calls_per_frame: {vlm_calls / frames_done:.3f}")
        calls_per_s = float(vlm_calls) / float(measure_elapsed_s)
        if calls_per_s > 0:
            target_ms = (1.0 / calls_per_s) * 1000.0
            print(f"- observed_dispatch_rate: {calls_per_s:.3f} calls/sec")
            print(f"- target_vlm_query_time_to_keep_up: <= {target_ms:.0f} ms/query at this dispatch rate (single-thread, synchronous)")
        print(
            "- interpretation: if you only need small JSON, reduce max_new_tokens/prompt length; "
            "for smooth real-time capture, run VLM async or keep dispatch_rate low enough that it never blocks the frame loop."
        )


def _git_branch() -> str:
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO_ROOT))
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()

