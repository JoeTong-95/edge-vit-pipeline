#!/usr/bin/env python3
"""
Truck-oriented VLM benchmark (samples 1–4, 60 s each).

Primary metric (all parts): **max non-lost truck+bus tracks per frame** — peak count of
tracks with detector class truck or bus and status in {new, active} within the measure window.
Also records **VLM completions** (successful inferences) and wall/GPU rates.

Parts:
  A — Baseline + microbatch sizes 1, 4, 8, 10 (fixed default crop_cache_size).
  B — microbatch 10, crop_cache_size (interval) ∈ {2, 4, 8, 12}.
  C — Re-run best config from A and B per sample (max vlm_completions, tie-break peak trucks/frame);
 compare sync completions/s vs synthetic JSONL cache drain at the same batch size.

Results: SQLite DB + PNG plots under --output-dir. MySQL DDL: vlm_truck_benchmark_schema.sql
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO_ROOT = _THIS.parent.parent
_CONFIG = _THIS.parent / "configuration-layer"
_INPUT = _THIS.parent / "input-layer"
_ROI = _THIS.parent / "roi-layer"
_YOLO = _THIS.parent / "yolo-layer"
_TRACK = _THIS.parent / "tracking-layer"
_VSTATE = _THIS.parent / "vehicle-state-layer"
_CROPPER = _THIS.parent / "vlm-frame-cropper-layer"

for p in (_THIS, _CONFIG, _INPUT, _ROI, _YOLO, _TRACK, _VSTATE, _CROPPER):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEFAULT_SAMPLE_VIDEOS = [f"data/sample{i}.mp4" for i in range(1, 5)]
DEFAULT_OUTPUT_DIR = Path(r"E:\OneDrive\desktop\vlm-layer")


def _resolve_repo_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())


def _safe_div(n: float, d: float) -> float:
    return n / d if d not in (0.0, -0.0) else 0.0


def _resolve_sample_video_paths(rel_paths: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for rel in rel_paths:
        abs_path = _resolve_repo_path(rel)
        label = Path(rel).stem
        if os.path.isfile(abs_path):
            out.append((label, abs_path))
        else:
            print(f"[truck_benchmark] skip missing video: {rel}", flush=True)
    return out


def _is_truck_bus_class(name: str) -> bool:
    n = name.lower()
    return n in ("truck", "bus")


def _run_truck_session(
    *,
    video_path: str,
    frame_resolution: tuple[int, int],
    device: str,
    yolo_model: str,
    yolo_conf: float,
    roi_enabled: bool,
    roi_threshold: int,
    vlm_model_resolved: str,
    vlm_crop_cache_size: int,
    vlm_dead_after_lost: int,
    vlm_batch_size: int,
    warmup_seconds: float,
    measure_seconds: float,
    measure_only_after_roi_lock: bool,
    vlm_state: Optional[Any] = None,
) -> tuple[dict[str, Any], Any]:
    from detector import build_yolo_layer_package, filter_yolo_detections, initialize_yolo_layer, run_yolo_detection
    from input_layer import InputLayer
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
    from layer import (
        VLMConfig,
        VLMFrameCropperLayerPackage,
        build_vlm_ack_package_from_result,
        initialize_vlm_layer,
        run_vlm_inference_batch,
    )

    vlm_query_type = "vehicle_semantics_single_shot_v1"
    vlm_batch_size = max(1, int(vlm_batch_size))

    input_layer = InputLayer()
    input_layer.initialize_input_layer(
        config_input_source="video",
        config_frame_resolution=frame_resolution,
        config_input_path=video_path,
    )
    initialize_roi_layer(config_roi_enabled=roi_enabled, config_roi_vehicle_count_threshold=roi_threshold)
    initialize_yolo_layer(model_name=yolo_model, conf_threshold=yolo_conf, device=device)
    initialize_tracking_layer(frame_rate=30)
    initialize_vehicle_state_layer(prune_after_lost_frames=None)

    if vlm_state is None:
        vlm_state = initialize_vlm_layer(
            VLMConfig(
                config_vlm_enabled=True,
                config_vlm_model=vlm_model_resolved,
                config_device=device,
            )
        )
    crop_cache = initialize_vlm_crop_cache(
        config_vlm_crop_cache_size=vlm_crop_cache_size,
        config_vlm_dead_after_lost_frames=vlm_dead_after_lost,
    )

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
    vlm_calls = 0
    vlm_query_s_total = 0.0
    vlm_query_count = 0
    roi_lock_frame_id: int | None = None
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

    max_nonlost_tb = 0
    sum_nonlost_tb = 0.0
    frames_tb = 0

    def process_one(record_truck: bool) -> bool:
        nonlocal det_counts, track_counts, vlm_calls, vlm_query_s_total, vlm_query_count
        nonlocal roi_lock_frame_id, roi_last_candidate_count, roi_last_locked, roi_last_bounds
        nonlocal yolo_full_s_total, yolo_full_frames, yolo_roi_s_total, yolo_roi_frames
        nonlocal yolo_full_infer_s_total, yolo_full_post_s_total, yolo_roi_infer_s_total, yolo_roi_post_s_total
        nonlocal yolo_full_pixels_total, yolo_roi_pixels_total, yolo_full_det_total, yolo_roi_det_total
        nonlocal max_nonlost_tb, sum_nonlost_tb, frames_tb
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
        yolo_infer_dt = t_inf1 - t_inf0
        yolo_post_dt = t_y1 - t_inf1
        if yolo_using_roi:
            yolo_roi_s_total += t_y1 - t_y0
            yolo_roi_frames += 1
            yolo_roi_infer_s_total += yolo_infer_dt
            yolo_roi_post_s_total += yolo_post_dt
            yolo_roi_det_total += len(dets)
        else:
            yolo_full_s_total += t_y1 - t_y0
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

        if record_truck:
            classes = tracking_pkg.get("tracking_layer_detector_class", [])
            statuses = tracking_pkg.get("tracking_layer_status", [])
            tb = 0
            for i in range(len(classes)):
                if _is_truck_bus_class(str(classes[i])) and str(statuses[i]) != "lost":
                    tb += 1
            max_nonlost_tb = max(max_nonlost_tb, tb)
            sum_nonlost_tb += float(tb)
            frames_tb += 1

        t_st0 = time.perf_counter()
        update_vehicle_state_from_tracking(tracking_pkg)
        t_st1 = time.perf_counter()

        t_v0 = time.perf_counter()
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
                vlm_frame_cropper_trigger_reason=f"truck_bench:{row['status']}",
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
                f"truck_bench:{row['status']}",
            )

        dispatch_pkgs: list[VLMFrameCropperLayerPackage] = []
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
            if len(dispatch_pkgs) >= vlm_batch_size:
                break

        if dispatch_pkgs:
            q0 = time.perf_counter()
            raw_results = run_vlm_inference_batch(
                vlm_state,
                dispatch_pkgs,
                [vlm_query_type] * len(dispatch_pkgs),
            )
            q1 = time.perf_counter()
            vlm_query_s_total += q1 - q0
            vlm_query_count += len(raw_results)
            for raw_result in raw_results:
                ack = build_vlm_ack_package_from_result(raw_result)
                register_vlm_ack_package(crop_cache, ack)
            vlm_calls += len(raw_results)
        t_v1 = time.perf_counter()

        t1 = time.perf_counter()
        totals["input_s"] += t_in1 - t_in0
        totals["roi_s"] += t_roi1 - t_roi0
        totals["yolo_s"] += t_y1 - t_y0
        totals["tracking_s"] += t_tr1 - t_tr0
        totals["state_s"] += t_st1 - t_st0
        totals["vlm_s"] += t_v1 - t_v0
        totals["scene_s"] += 0.0
        totals["end_to_end_s"] += t1 - t0
        return True

    t_warm_end = time.perf_counter() + warmup_seconds
    while time.perf_counter() < t_warm_end:
        if not process_one(False):
            break

    if roi_enabled and measure_only_after_roi_lock:
        while roi_lock_frame_id is None:
            if not process_one(False):
                break
            if time.perf_counter() > t_warm_end + 30.0:
                break

    for k in totals:
        totals[k] = 0.0
    det_counts = 0
    track_counts = 0
    vlm_calls = 0
    vlm_query_s_total = 0.0
    vlm_query_count = 0
    roi_lock_frame_id = None
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
    max_nonlost_tb = 0
    sum_nonlost_tb = 0.0
    frames_tb = 0

    frames_done = 0
    measure_started = time.perf_counter()
    t_measure_end = time.perf_counter() + measure_seconds
    while time.perf_counter() < t_measure_end:
        if not process_one(True):
            break
        frames_done += 1
    measure_elapsed_s = max(1e-9, time.perf_counter() - measure_started)

    input_layer.close_input_layer()

    if frames_done <= 0:
        raise RuntimeError("No frames processed in measured section.")

    pipeline_fps = _safe_div(float(frames_done), totals["end_to_end_s"])
    avg_q_ms = (_safe_div(vlm_query_s_total, float(vlm_query_count)) * 1000.0) if vlm_query_count > 0 else 0.0
    gpu_qps = _safe_div(float(vlm_query_count), vlm_query_s_total) if vlm_query_s_total > 0 else 0.0
    trucks_wall = _safe_div(float(vlm_calls), measure_elapsed_s)
    avg_tb = _safe_div(sum_nonlost_tb, float(frames_tb)) if frames_tb > 0 else 0.0

    return (
        {
            "frames_done": frames_done,
            "measure_elapsed_s": measure_elapsed_s,
            "vlm_calls": vlm_calls,
            "vlm_query_count": vlm_query_count,
            "vlm_query_s_total": vlm_query_s_total,
            "pipeline_fps": pipeline_fps,
            "vlm_per_sec_wall": trucks_wall,
            "gpu_queries_per_sec": gpu_qps,
            "avg_ms_per_vlm_query": avg_q_ms,
            "vlm_batch_size": vlm_batch_size,
            "crop_cache_size": vlm_crop_cache_size,
            "max_nonlost_truck_bus_per_frame": float(max_nonlost_tb),
            "avg_nonlost_truck_bus_per_frame": float(avg_tb),
        },
        vlm_state,
    )


def _run_cache_flow_jsonl_drain(
    vlm_state: Any,
    *,
    batch_size: int,
    num_tasks: int,
    query_type: str,
    jsonl_path: Path,
) -> dict[str, Any]:
    from layer import VLMFrameCropperLayerPackage, run_vlm_inference_batch
    from vlm_deferred_queue import (
        DeferredVLMTask,
        append_deferred_task,
        decode_crop_image,
        encode_crop_image_to_png_base64,
        load_deferred_tasks,
    )

    if jsonl_path.exists():
        jsonl_path.unlink()
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(int(num_tasks)):
        crop = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        enc = encode_crop_image_to_png_base64(crop)
        append_deferred_task(
            jsonl_path,
            DeferredVLMTask(
                track_id=str(i),
                dispatch_frame_id=i,
                query_type=query_type,
                prompt_text="",
                crop_png_base64=enc,
                bbox=(0, 0, 128, 128),
                created_at_unix_s=time.time(),
            ),
            max_file_bytes=None,
        )
    tasks = load_deferred_tasks(jsonl_path)
    if not tasks:
        raise RuntimeError("cache-flow: no tasks loaded from JSONL")

    bs = max(1, int(batch_size))
    wall0 = time.perf_counter()
    gpu_s = 0.0
    total_q = 0
    for base in range(0, len(tasks), bs):
        batch = tasks[base : base + bs]
        pkgs = [
            VLMFrameCropperLayerPackage(
                vlm_frame_cropper_layer_track_id=t.track_id,
                vlm_frame_cropper_layer_image=decode_crop_image(t.crop_png_base64),
                vlm_frame_cropper_layer_bbox=t.bbox if t.bbox is not None else (0, 0, 128, 128),
            )
            for t in batch
        ]
        q0 = time.perf_counter()
        _ = run_vlm_inference_batch(vlm_state, pkgs, [query_type] * len(pkgs))
        gpu_s += time.perf_counter() - q0
        total_q += len(pkgs)
    wall_s = max(1e-9, time.perf_counter() - wall0)

    return {
        "vlm_calls": total_q,
        "measure_elapsed_s": wall_s,
        "vlm_query_s_total": gpu_s,
        "cache_drain_jobs_per_sec": _safe_div(float(total_q), wall_s),
    }


def _init_sqlite(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS vlm_truck_benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            part TEXT NOT NULL,
            sample_label TEXT NOT NULL,
            video_path TEXT NOT NULL,
            vlm_batch_size INTEGER NOT NULL,
            crop_cache_size INTEGER NOT NULL,
            measure_seconds REAL NOT NULL,
            warmup_seconds REAL NOT NULL,
            frames_done INTEGER NOT NULL,
            max_nonlost_truck_bus_per_frame REAL NOT NULL,
            avg_nonlost_truck_bus_per_frame REAL NOT NULL,
            vlm_completions INTEGER NOT NULL,
            vlm_per_sec_wall REAL NOT NULL,
            gpu_queries_per_sec REAL NOT NULL,
            pipeline_fps REAL NOT NULL,
            sync_vlm_per_sec REAL,
            cache_drain_jobs_per_sec REAL,
            cache_uplift_ratio REAL,
            extra_json TEXT
        )
        """
    )
    conn.commit()
    return conn


def _insert_row(
    conn: sqlite3.Connection,
    row: dict[str, Any],
) -> None:
    keys = [
        "created_at",
        "part",
        "sample_label",
        "video_path",
        "vlm_batch_size",
        "crop_cache_size",
        "measure_seconds",
        "warmup_seconds",
        "frames_done",
        "max_nonlost_truck_bus_per_frame",
        "avg_nonlost_truck_bus_per_frame",
        "vlm_completions",
        "vlm_per_sec_wall",
        "gpu_queries_per_sec",
        "pipeline_fps",
        "sync_vlm_per_sec",
        "cache_drain_jobs_per_sec",
        "cache_uplift_ratio",
        "extra_json",
    ]
    vals = [row.get(k) for k in keys]
    placeholders = ",".join("?" * len(keys))
    conn.execute(
        f"INSERT INTO vlm_truck_benchmark_runs ({','.join(keys)}) VALUES ({placeholders})",
        vals,
    )
    conn.commit()


def _score_run(r: dict[str, Any]) -> tuple[float, float]:
    return (float(r["vlm_calls"]), float(r["max_nonlost_truck_bus_per_frame"]))


def _plot_three_panels(
    *,
    samples: list[str],
    part_a: dict[str, dict[int, dict[str, Any]]],
    part_b: dict[str, dict[int, dict[str, Any]]],
    part_c: dict[str, dict[str, Any]],
    output_png: Path,
) -> None:
    BACKGROUND = "#171717"
    PANEL = "#242424"
    GRID = "#4a4a4a"
    TEXT = "#f3f3f3"

    batch_keys_a = sorted({bk for per in part_a.values() for bk in per.keys()})
    interval_keys_b = sorted({ik for per in part_b.values() for ik in per.keys()})

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    fig.patch.set_facecolor(BACKGROUND)

    def style_ax(ax: Any) -> None:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT)
        ax.grid(True, axis="y", color=GRID, alpha=0.35)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.set_ylabel("max truck+bus / frame\n(non-lost)", color=TEXT, fontsize=9)

    x = np.arange(len(samples))
    w = min(0.2, 0.8 / max(len(batch_keys_a), 1))
    ax = axes[0]
    style_ax(ax)
    for i, b in enumerate(batch_keys_a):
        ys = [part_a[s].get(b, {}).get("max_nonlost_truck_bus_per_frame", 0.0) for s in samples]
        ax.bar(x + (i - len(batch_keys_a) / 2) * w + w / 2, ys, width=w, label=f"B={b}")
    ax.set_xticks(x)
    ax.set_xticklabels(samples, color=TEXT)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    ax.set_title("A: microbatch (baseline B=1, 4, 8, 10)", color=TEXT)

    w2 = min(0.2, 0.8 / max(len(interval_keys_b), 1))
    ax = axes[1]
    style_ax(ax)
    for i, iv in enumerate(interval_keys_b):
        ys = [part_b[s].get(iv, {}).get("max_nonlost_truck_bus_per_frame", 0.0) for s in samples]
        ax.bar(x + (i - len(interval_keys_b) / 2) * w2 + w2 / 2, ys, width=w2, label=f"cache={iv}")
    ax.set_xticks(x)
    ax.set_xticklabels(samples, color=TEXT)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    ax.set_title("B: microbatch=10, crop cache size (interval) 2–12", color=TEXT)

    ax = axes[2]
    style_ax(ax)
    sync_y = [part_c[s].get("max_nonlost_truck_bus_per_frame", 0.0) for s in samples]
    ax.bar(x - 0.18, sync_y, width=0.35, label="sync best cfg (peak trucks/frame)")
    drain_y = [part_c[s].get("drain_equiv_peak", 0.0) for s in samples]
    ax.bar(x + 0.18, drain_y, width=0.35, label="cache drain (synthetic; not same workload)")
    ax.set_xticks(x)
    ax.set_xticklabels(samples, color=TEXT)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    ax.set_title("C: best A/B config — sync peak vs scaled drain (see summary JSON)", color=TEXT)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160, facecolor=BACKGROUND)
    plt.close(fig)


def _clear_output_dir(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description="Truck VLM benchmark A/B/C + SQLite + plots.")
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--measure-seconds", type=float, default=60.0)
    ap.add_argument("--warmup-seconds", type=float, default=10.0)
    ap.add_argument("--device", type=str, default=None, help="Override config_device (default: config.yaml)")
    ap.add_argument("--yolo-model", type=str, default=None)
    ap.add_argument("--yolo-conf", type=float, default=None)
    ap.add_argument("--vlm-model", type=str, default=None, help="Repo-relative model dir; default config_vlm_model")
    ap.add_argument("--frame-width", type=int, default=None, help="With --frame-height, override config_frame_resolution (width, height)")
    ap.add_argument("--frame-height", type=int, default=None)
    roi_grp = ap.add_mutually_exclusive_group()
    roi_grp.add_argument("--roi-enabled", action="store_true", help="Force ROI on")
    roi_grp.add_argument("--roi-disabled", action="store_true", help="Force ROI off")
    ap.add_argument("--roi-threshold", type=int, default=None)
    ap.add_argument("--vlm-dead-after-lost", type=int, default=None)
    ap.add_argument(
        "--default-crop-cache",
        type=int,
        default=None,
        help="Part A fixed crop_cache_size (default: 4, or config_vlm_crop_cache_size if you prefer — we use 4 for A unless set)",
    )
    ap.add_argument("--cache-drain-tasks", type=int, default=64)
    ap.add_argument("--skip-clear", action="store_true", help="Do not delete existing output-dir contents")
    args = ap.parse_args()

    from config_node import get_config_value, load_config, validate_config

    cfg = load_config(_CONFIG / "config.yaml")
    validate_config(cfg)
    if str(get_config_value(cfg, "config_input_source")) != "video":
        print("[truck_benchmark] config_input_source must be video.", flush=True)
        return 1
    if not bool(get_config_value(cfg, "config_vlm_enabled")):
        print("[truck_benchmark] Set config_vlm_enabled=true.", flush=True)
        return 1

    device = args.device if args.device is not None else str(get_config_value(cfg, "config_device"))
    yolo_model = args.yolo_model if args.yolo_model is not None else str(get_config_value(cfg, "config_yolo_model"))
    yolo_conf = float(args.yolo_conf if args.yolo_conf is not None else get_config_value(cfg, "config_yolo_confidence_threshold"))
    if args.roi_disabled:
        roi_enabled = False
    elif args.roi_enabled:
        roi_enabled = True
    else:
        roi_enabled = bool(get_config_value(cfg, "config_roi_enabled"))
    roi_threshold = int(args.roi_threshold if args.roi_threshold is not None else get_config_value(cfg, "config_roi_vehicle_count_threshold"))
    vlm_dead_after_lost = int(
        args.vlm_dead_after_lost if args.vlm_dead_after_lost is not None else get_config_value(cfg, "config_vlm_dead_after_lost_frames")
    )
    vlm_rel = args.vlm_model if args.vlm_model is not None else str(get_config_value(cfg, "config_vlm_model"))
    vlm_model_resolved = _resolve_repo_path(vlm_rel)
    if args.frame_width is not None and args.frame_height is not None:
        frame_resolution = (int(args.frame_width), int(args.frame_height))
    elif args.frame_width is not None or args.frame_height is not None:
        print("[truck_benchmark] Pass both --frame-width and --frame-height or neither.", flush=True)
        return 1
    else:
        frame_resolution = tuple(int(x) for x in get_config_value(cfg, "config_frame_resolution"))
    part_a_crop_cache = int(args.default_crop_cache if args.default_crop_cache is not None else 4)

    out = Path(args.output_dir)
    if not args.skip_clear:
        _clear_output_dir(out)

    samples = _resolve_sample_video_paths(DEFAULT_SAMPLE_VIDEOS)
    if not samples:
        print("[truck_benchmark] No sample videos found.", flush=True)
        return 1

    sample_labels = [s[0] for s in samples]
    conn = _init_sqlite(out / "vlm_truck_benchmark.sqlite")
    created = datetime.now(timezone.utc).isoformat()

    part_a: dict[str, dict[int, dict[str, Any]]] = {sl: {} for sl in sample_labels}
    part_b: dict[str, dict[int, dict[str, Any]]] = {sl: {} for sl in sample_labels}

    vlm_state = None

    # --- Part A ---
    for label, video in samples:
        for b in (1, 4, 8, 10):
            print(f"[A] {label} batch={b} crop_cache={part_a_crop_cache}", flush=True)
            stats, vlm_state = _run_truck_session(
                video_path=video,
                frame_resolution=frame_resolution,
                device=device,
                yolo_model=yolo_model,
                yolo_conf=yolo_conf,
                roi_enabled=roi_enabled,
                roi_threshold=roi_threshold,
                vlm_model_resolved=vlm_model_resolved,
                vlm_crop_cache_size=part_a_crop_cache,
                vlm_dead_after_lost=vlm_dead_after_lost,
                vlm_batch_size=b,
                warmup_seconds=args.warmup_seconds,
                measure_seconds=args.measure_seconds,
                measure_only_after_roi_lock=roi_enabled,
                vlm_state=vlm_state,
            )
            part_a[label][b] = stats
            extra = json.dumps({"part": "A", "microbatch": b})
            _insert_row(
                conn,
                {
                    "created_at": created,
                    "part": "A",
                    "sample_label": label,
                    "video_path": video,
                    "vlm_batch_size": b,
                    "crop_cache_size": part_a_crop_cache,
                    "measure_seconds": args.measure_seconds,
                    "warmup_seconds": args.warmup_seconds,
                    "frames_done": stats["frames_done"],
                    "max_nonlost_truck_bus_per_frame": stats["max_nonlost_truck_bus_per_frame"],
                    "avg_nonlost_truck_bus_per_frame": stats["avg_nonlost_truck_bus_per_frame"],
                    "vlm_completions": stats["vlm_calls"],
                    "vlm_per_sec_wall": stats["vlm_per_sec_wall"],
                    "gpu_queries_per_sec": stats["gpu_queries_per_sec"],
                    "pipeline_fps": stats["pipeline_fps"],
                    "sync_vlm_per_sec": None,
                    "cache_drain_jobs_per_sec": None,
                    "cache_uplift_ratio": None,
                    "extra_json": extra,
                },
            )

    # --- Part B ---
    for label, video in samples:
        for cache_sz in (2, 4, 8, 12):
            print(f"[B] {label} batch=10 crop_cache={cache_sz}", flush=True)
            stats, vlm_state = _run_truck_session(
                video_path=video,
                frame_resolution=frame_resolution,
                device=device,
                yolo_model=yolo_model,
                yolo_conf=yolo_conf,
                roi_enabled=roi_enabled,
                roi_threshold=roi_threshold,
                vlm_model_resolved=vlm_model_resolved,
                vlm_crop_cache_size=cache_sz,
                vlm_dead_after_lost=vlm_dead_after_lost,
                vlm_batch_size=10,
                warmup_seconds=args.warmup_seconds,
                measure_seconds=args.measure_seconds,
                measure_only_after_roi_lock=roi_enabled,
                vlm_state=vlm_state,
            )
            part_b[label][cache_sz] = stats
            _insert_row(
                conn,
                {
                    "created_at": created,
                    "part": "B",
                    "sample_label": label,
                    "video_path": video,
                    "vlm_batch_size": 10,
                    "crop_cache_size": cache_sz,
                    "measure_seconds": args.measure_seconds,
                    "warmup_seconds": args.warmup_seconds,
                    "frames_done": stats["frames_done"],
                    "max_nonlost_truck_bus_per_frame": stats["max_nonlost_truck_bus_per_frame"],
                    "avg_nonlost_truck_bus_per_frame": stats["avg_nonlost_truck_bus_per_frame"],
                    "vlm_completions": stats["vlm_calls"],
                    "vlm_per_sec_wall": stats["vlm_per_sec_wall"],
                    "gpu_queries_per_sec": stats["gpu_queries_per_sec"],
                    "pipeline_fps": stats["pipeline_fps"],
                    "sync_vlm_per_sec": None,
                    "cache_drain_jobs_per_sec": None,
                    "cache_uplift_ratio": None,
                    "extra_json": json.dumps({"part": "B", "crop_interval": cache_sz}),
                },
            )

    # --- Pick best per sample from A ∪ B ---
    best_by_sample: dict[str, dict[str, Any]] = {}
    for label in sample_labels:
        candidates: list[dict[str, Any]] = []
        for b, st in part_a[label].items():
            candidates.append({**st, "_tag": f"A_B{b}", "vlm_batch_size": b, "crop_cache_size": part_a_crop_cache})
        for c, st in part_b[label].items():
            candidates.append({**st, "_tag": f"B_cache{c}", "vlm_batch_size": 10, "crop_cache_size": c})
        best = max(candidates, key=_score_run)
        best_by_sample[label] = best
        print(
            f"[best] {label} -> {best['_tag']} vlm_calls={best['vlm_calls']} peak_tb/frame={best['max_nonlost_truck_bus_per_frame']}",
            flush=True,
        )

    # --- Part C ---
    part_c: dict[str, dict[str, Any]] = {}
    for label, video in samples:
        bcfg = best_by_sample[label]
        bsz = int(bcfg["vlm_batch_size"])
        csz = int(bcfg["crop_cache_size"])
        print(f"[C sync] {label} best batch={bsz} cache={csz}", flush=True)
        stats, vlm_state = _run_truck_session(
            video_path=video,
            frame_resolution=frame_resolution,
            device=device,
            yolo_model=yolo_model,
            yolo_conf=yolo_conf,
            roi_enabled=roi_enabled,
            roi_threshold=roi_threshold,
            vlm_model_resolved=vlm_model_resolved,
            vlm_crop_cache_size=csz,
            vlm_dead_after_lost=vlm_dead_after_lost,
            vlm_batch_size=bsz,
            warmup_seconds=args.warmup_seconds,
            measure_seconds=args.measure_seconds,
            measure_only_after_roi_lock=roi_enabled,
            vlm_state=vlm_state,
        )
        jsonl_path = out / f"cache_flow_{label}.jsonl"
        drain = _run_cache_flow_jsonl_drain(
            vlm_state,
            batch_size=bsz,
            num_tasks=args.cache_drain_tasks,
            query_type="vehicle_semantics_single_shot_v1",
            jsonl_path=jsonl_path,
        )
        sync_rate = stats["vlm_per_sec_wall"]
        drain_rate = drain["cache_drain_jobs_per_sec"]
        ratio = _safe_div(drain_rate, sync_rate) if sync_rate > 0 else 0.0
        # For panel C second bar: show relative throughput as synthetic "equivalent peak" scale
        drain_equiv_peak = stats["max_nonlost_truck_bus_per_frame"] * ratio
        part_c[label] = {
            "max_nonlost_truck_bus_per_frame": stats["max_nonlost_truck_bus_per_frame"],
            "drain_equiv_peak": drain_equiv_peak,
            "sync_vlm_per_sec": sync_rate,
            "cache_drain_jobs_per_sec": drain_rate,
            "cache_uplift_ratio": ratio,
            "best_tag": bcfg["_tag"],
        }
        _insert_row(
            conn,
            {
                "created_at": created,
                "part": "C",
                "sample_label": label,
                "video_path": video,
                "vlm_batch_size": bsz,
                "crop_cache_size": csz,
                "measure_seconds": args.measure_seconds,
                "warmup_seconds": args.warmup_seconds,
                "frames_done": stats["frames_done"],
                "max_nonlost_truck_bus_per_frame": stats["max_nonlost_truck_bus_per_frame"],
                "avg_nonlost_truck_bus_per_frame": stats["avg_nonlost_truck_bus_per_frame"],
                "vlm_completions": stats["vlm_calls"],
                "vlm_per_sec_wall": stats["vlm_per_sec_wall"],
                "gpu_queries_per_sec": stats["gpu_queries_per_sec"],
                "pipeline_fps": stats["pipeline_fps"],
                "sync_vlm_per_sec": sync_rate,
                "cache_drain_jobs_per_sec": drain_rate,
                "cache_uplift_ratio": ratio,
                "extra_json": json.dumps(
                    {
                        "best_from_ab": bcfg["_tag"],
                        "note": "drain is synthetic JSONL batching, not live truck workload",
                    }
                ),
            },
        )

    summary = {
        "samples": sample_labels,
        "part_a": {k: {str(b): v for b, v in part_a[k].items()} for k in sample_labels},
        "part_b": {k: {str(i): v for i, v in part_b[k].items()} for k in sample_labels},
        "part_c": part_c,
        "metric_docs": {
            "max_nonlost_truck_bus_per_frame": "Max simultaneous truck+bus tracks (new/active) in one frame during measure window.",
            "vlm_completions": "Total VLM inferences completed in measure window.",
        },
    }
    (out / "truck_benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_three_panels(
        samples=sample_labels,
        part_a=part_a,
        part_b=part_b,
        part_c=part_c,
        output_png=out / "truck_benchmark_ABC.png",
    )

    print(f"[truck_benchmark] Done. DB={out / 'vlm_truck_benchmark.sqlite'} plot={out / 'truck_benchmark_ABC.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
