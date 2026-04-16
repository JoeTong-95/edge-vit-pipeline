#!/usr/bin/env python3
"""
measure_vlm_modes.py

Produces **three** comparable modes (PNG + JSON):

1. **Baseline** — full video pipeline, synchronous VLM, micro-batch size 1.
2. **Microbatching** — same pipeline, larger micro-batch (fewer `generate()` calls per crop set).
3. **Cache flow** — spill-style path: crops serialized like `vlm_deferred_queue` JSONL, then
   batched offline drain (same `run_vlm_inference_batch` as `run_deferred_vlm_queue.py`).

Separate from evaluation-output-layer `benchmark.py` (single-purpose device profile).

Examples:
  python measure_vlm_modes.py
  python measure_vlm_modes.py --microbatch-size 4 --cache-drain-batch 8 --output-dir "D:\\runs\\vlm-modes"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

_THIS = Path(__file__).resolve().parent
_VLM = _THIS.parent
_SRC = _VLM.parent
_REPO_ROOT = _SRC.parent
_CONFIG = _SRC / "configuration-layer"
_INPUT = _SRC / "input-layer"
_ROI = _SRC / "roi-layer"
_YOLO = _SRC / "yolo-layer"
_TRACK = _SRC / "tracking-layer"
_VSTATE = _SRC / "vehicle-state-layer"
_CROPPER = _SRC / "vlm-frame-cropper-layer"
_VLM_UTIL = _VLM / "util"

for p in (_THIS, _VLM, _VLM_UTIL, _CONFIG, _INPUT, _ROI, _YOLO, _TRACK, _VSTATE, _CROPPER):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Default multi-sample set (repo-relative); missing files are skipped with a warning.
DEFAULT_SAMPLE_VIDEOS = [f"data/sample{i}.mp4" for i in range(1, 5)]


def _resolve_repo_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())


def _safe_div(n: float, d: float) -> float:
    return n / d if d not in (0.0, -0.0) else 0.0


def _resolve_sample_video_paths(rel_paths: list[str]) -> list[tuple[str, str]]:
    """Return [(label, absolute_path), ...] for files that exist."""
    out: list[tuple[str, str]] = []
    for rel in rel_paths:
        abs_path = _resolve_repo_path(rel)
        label = Path(rel).stem
        if os.path.isfile(abs_path):
            out.append((label, abs_path))
        else:
            print(f"[measure_vlm_modes] skip missing video: {rel}", flush=True)
    return out


def _run_one_pipeline_session(
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
        nonlocal det_counts, track_counts, vlm_calls, vlm_query_s_total, vlm_query_count
        nonlocal roi_lock_frame_id, roi_lock_candidate_count, roi_lock_bounds
        nonlocal roi_last_candidate_count, roi_last_locked, roi_last_bounds
        nonlocal yolo_full_s_total, yolo_full_frames, yolo_roi_s_total, yolo_roi_frames
        nonlocal yolo_full_infer_s_total, yolo_full_post_s_total, yolo_roi_infer_s_total, yolo_roi_post_s_total
        nonlocal yolo_full_pixels_total, yolo_roi_pixels_total, yolo_full_det_total, yolo_roi_det_total
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
                vlm_frame_cropper_trigger_reason=f"mode_cmp:{row['status']}",
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
                f"mode_cmp:{row['status']}",
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

    # Warmup
    t_warm_end = time.perf_counter() + warmup_seconds
    while time.perf_counter() < t_warm_end:
        if not process_one():
            break

    roi_waited_s = 0.0
    if roi_enabled and measure_only_after_roi_lock:
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
    t_measure_end = time.perf_counter() + measure_seconds
    while time.perf_counter() < t_measure_end:
        if not process_one():
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

    return (
        {
            "frames_done": frames_done,
            "measure_elapsed_s": measure_elapsed_s,
            "vlm_calls": vlm_calls,
            "vlm_query_count": vlm_query_count,
            "vlm_query_s_total": vlm_query_s_total,
            "pipeline_fps": pipeline_fps,
            "trucks_per_sec_wall": trucks_wall,
            "vlm_queries_per_gpu_sec": gpu_qps,
            "avg_ms_per_vlm_query": avg_q_ms,
            "roi_waited_s": roi_waited_s,
            "roi_locked_for_measure": roi_lock_frame_id is not None,
            "vlm_batch_size": vlm_batch_size,
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
    """
    Measure spill/cache path: write JSONL like deferred queue, load, batch-infer (offline drain).
    """
    import numpy as np

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

    avg_q_ms = (gpu_s / float(total_q) * 1000.0) if total_q > 0 else 0.0
    return {
        "mode": "cache_flow_jsonl_drain",
        "vlm_batch_size": bs,
        "frames_done": 0,
        "measure_elapsed_s": wall_s,
        "vlm_calls": total_q,
        "vlm_query_count": total_q,
        "vlm_query_s_total": gpu_s,
        "pipeline_fps": 0.0,
        "trucks_per_sec_wall": _safe_div(float(total_q), wall_s),
        "vlm_queries_per_gpu_sec": _safe_div(float(total_q), gpu_s) if gpu_s > 0 else 0.0,
        "avg_ms_per_vlm_query": avg_q_ms,
        "cache_drain_jobs_per_sec": _safe_div(float(total_q), wall_s),
    }


def _plot_grouped(
    *,
    sample_labels: list[str],
    mode_titles: list[str],
    trucks: list[list[float]],
    gpu_qps: list[list[float]],
    panel3: list[list[float]],
    output_png: Path,
    title: str,
) -> None:
    """One figure, three subplots. Rows = modes (baseline, microbatch, cache); cols = samples (grouped bars)."""
    BACKGROUND = "#171717"
    PANEL = "#242424"
    GRID = "#4a4a4a"
    TEXT = "#f3f3f3"
    COLORS = ["#7cb6ff", "#ffb347", "#95e1d3", "#c792ea"]

    n_modes = len(mode_titles)
    n_s = len(sample_labels)
    x = np.arange(n_modes, dtype=float)
    width = min(0.22, 0.8 / max(n_s, 1))

    plt.style.use("default")
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.patch.set_facecolor(BACKGROUND)

    def style_ax(ax: Any) -> None:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT)
        ax.grid(True, axis="y", color=GRID, alpha=0.35)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    data_sets = [trucks, gpu_qps, panel3]
    ylabels = [
        "Dispatches or drain jobs / s (wall)",
        "VLM queries / s (GPU time)",
        "Video FPS | JSONL drain jobs/s (cache row)",
    ]
    subtitles = [
        "Wall throughput by sample × mode",
        "GPU throughput by sample × mode",
        "Pipeline FPS (baseline & microbatch) | drain jobs/s (cache; same for all samples)",
    ]

    for ax, mat, ylab, sub in zip(axes, data_sets, ylabels, subtitles, strict=True):
        style_ax(ax)
        for si in range(n_s):
            offset = (si - (n_s - 1) / 2.0) * width
            heights = [mat[m][si] for m in range(n_modes)]
            bars = ax.bar(
                x + offset,
                heights,
                width,
                label=sample_labels[si],
                color=COLORS[si % len(COLORS)],
                edgecolor=TEXT,
                linewidth=0.4,
            )
            for bar, v in zip(bars, heights, strict=True):
                if v <= 0:
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=TEXT,
                )
        ax.set_ylabel(ylab, color=TEXT, fontsize=9)
        ax.set_title(sub, color=TEXT, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(mode_titles, color=TEXT, fontsize=10)

    axes[0].legend(
        loc="upper right",
        fontsize=8,
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=TEXT,
    )
    fig.suptitle(title, color=TEXT, fontsize=13, y=0.98)
    MUTED = "#bdbdbd"
    fig.text(
        0.5,
        0.02,
        "Baseline / Microbatch: one run per sample video. Cache flow: one JSONL drain (video-independent); bars repeat the same value per sample for color legend.",
        ha="center",
        va="bottom",
        color=MUTED,
        fontsize=7,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-way compare: baseline (B=1), microbatching (B>1), cache flow (JSONL drain)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:\OneDrive\desktop\vlm-layer"),
        help="Directory for PNG + JSON (created if missing).",
    )
    parser.add_argument("--warmup-seconds", type=float, default=8.0)
    parser.add_argument("--measure-seconds", type=float, default=45.0)
    parser.add_argument("--microbatch-size", type=int, default=4, help="Microbatching run: max dispatches per VLM call (>=2).")
    parser.add_argument("--cache-drain-batch", type=int, default=8, help="Cache-flow run: batch size when draining JSONL.")
    parser.add_argument("--cache-drain-tasks", type=int, default=48, help="Number of synthetic crops written to JSONL for drain test.")
    parser.add_argument("--skip-roi-wait", action="store_true", help="Measure immediately (may dilute ROI if lock is late).")
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="Also run extra batch sizes for JSON only (not plotted); uses first resolved video.",
    )
    parser.add_argument(
        "--videos",
        type=str,
        nargs="*",
        default=None,
        help="Repo-relative video paths (default: data/sample1.mp4 … sample4.mp4).",
    )
    args = parser.parse_args()

    from config_node import get_config_value, load_config, validate_config

    cfg = load_config(_CONFIG / "config.yaml")
    validate_config(cfg)

    if str(get_config_value(cfg, "config_input_source")) != "video":
        raise RuntimeError("measure_vlm_modes.py requires config_input_source=video.")
    if not bool(get_config_value(cfg, "config_vlm_enabled")):
        raise RuntimeError("Set config_vlm_enabled=true to compare VLM modes.")

    rel_list = list(args.videos) if args.videos else list(DEFAULT_SAMPLE_VIDEOS)
    resolved = _resolve_sample_video_paths(rel_list)
    if not resolved:
        raise FileNotFoundError(
            f"No sample videos found. Tried: {rel_list}. Place data/sample1.mp4 … sample4.mp4 under the repo or pass --videos."
        )

    frame_resolution = tuple(get_config_value(cfg, "config_frame_resolution"))
    device = str(get_config_value(cfg, "config_device"))
    yolo_model = str(get_config_value(cfg, "config_yolo_model"))
    yolo_conf = float(get_config_value(cfg, "config_yolo_confidence_threshold"))
    roi_enabled = bool(get_config_value(cfg, "config_roi_enabled"))
    roi_threshold = int(get_config_value(cfg, "config_roi_vehicle_count_threshold"))
    vlm_model = str(get_config_value(cfg, "config_vlm_model"))
    vlm_model_resolved = _resolve_repo_path(vlm_model)
    vlm_crop_cache_size = int(get_config_value(cfg, "config_vlm_crop_cache_size"))
    vlm_dead_after_lost = int(get_config_value(cfg, "config_vlm_dead_after_lost_frames"))
    measure_after_roi = not args.skip_roi_wait

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"vlm_mode_comparison_{ts}.json"
    out_png = out_dir / f"vlm_mode_comparison_{ts}.png"

    shared_vlm: Any | None = None
    query_type = "vehicle_semantics_single_shot_v1"
    mb = max(2, int(args.microbatch_size))

    per_sample: list[dict[str, Any]] = []
    baseline_trucks: list[float] = []
    micro_trucks: list[float] = []
    baseline_gpu: list[float] = []
    micro_gpu: list[float] = []
    baseline_fps: list[float] = []
    micro_fps: list[float] = []

    sample_labels: list[str] = []
    for label, video_path in resolved:
        sample_labels.append(label)
        print(f"[measure_vlm_modes] {label}: baseline (B=1) ...", flush=True)
        stats_b, shared_vlm = _run_one_pipeline_session(
            video_path=video_path,
            frame_resolution=frame_resolution,
            device=device,
            yolo_model=yolo_model,
            yolo_conf=yolo_conf,
            roi_enabled=roi_enabled,
            roi_threshold=roi_threshold,
            vlm_model_resolved=vlm_model_resolved,
            vlm_crop_cache_size=vlm_crop_cache_size,
            vlm_dead_after_lost=vlm_dead_after_lost,
            vlm_batch_size=1,
            warmup_seconds=float(args.warmup_seconds),
            measure_seconds=float(args.measure_seconds),
            measure_only_after_roi_lock=measure_after_roi,
            vlm_state=shared_vlm,
        )
        print(f"[measure_vlm_modes] {label}: microbatch (B={mb}) ...", flush=True)
        stats_m, shared_vlm = _run_one_pipeline_session(
            video_path=video_path,
            frame_resolution=frame_resolution,
            device=device,
            yolo_model=yolo_model,
            yolo_conf=yolo_conf,
            roi_enabled=roi_enabled,
            roi_threshold=roi_threshold,
            vlm_model_resolved=vlm_model_resolved,
            vlm_crop_cache_size=vlm_crop_cache_size,
            vlm_dead_after_lost=vlm_dead_after_lost,
            vlm_batch_size=mb,
            warmup_seconds=float(args.warmup_seconds),
            measure_seconds=float(args.measure_seconds),
            measure_only_after_roi_lock=measure_after_roi,
            vlm_state=shared_vlm,
        )
        per_sample.append(
            {
                "label": label,
                "video_path": video_path,
                "baseline": stats_b,
                "microbatch": stats_m,
            }
        )
        baseline_trucks.append(float(stats_b["trucks_per_sec_wall"]))
        micro_trucks.append(float(stats_m["trucks_per_sec_wall"]))
        baseline_gpu.append(float(stats_b["vlm_queries_per_gpu_sec"]))
        micro_gpu.append(float(stats_m["vlm_queries_per_gpu_sec"]))
        baseline_fps.append(float(stats_b["pipeline_fps"]))
        micro_fps.append(float(stats_m["pipeline_fps"]))

    drain_jsonl = out_dir / f"_measure_vlm_modes_cache_{ts}.jsonl"
    print("[measure_vlm_modes] Cache flow (JSONL drain, once) ...", flush=True)
    stats_c = _run_cache_flow_jsonl_drain(
        shared_vlm,
        batch_size=int(args.cache_drain_batch),
        num_tasks=int(args.cache_drain_tasks),
        query_type=query_type,
        jsonl_path=drain_jsonl,
    )
    try:
        drain_jsonl.unlink(missing_ok=True)
    except OSError:
        pass

    c_wall = float(stats_c["trucks_per_sec_wall"])
    c_gpu = float(stats_c["vlm_queries_per_gpu_sec"])
    c_drain = float(stats_c.get("cache_drain_jobs_per_sec", c_wall))
    n = len(sample_labels)
    cache_trucks = [c_wall] * n
    cache_gpu = [c_gpu] * n
    cache_panel3 = [c_drain] * n

    trucks_mat = [baseline_trucks, micro_trucks, cache_trucks]
    gpu_mat = [baseline_gpu, micro_gpu, cache_gpu]
    panel3_mat = [baseline_fps, micro_fps, cache_panel3]

    extra_runs: list[dict[str, Any]] = []
    if args.full_sweep and resolved:
        first_video = resolved[0][1]
        for b in (2, 4, 8):
            if b == mb or b == 1:
                continue
            print(f"[measure_vlm_modes] (extra) micro_batch_size={b} ...", flush=True)
            st, shared_vlm = _run_one_pipeline_session(
                video_path=first_video,
                frame_resolution=frame_resolution,
                device=device,
                yolo_model=yolo_model,
                yolo_conf=yolo_conf,
                roi_enabled=roi_enabled,
                roi_threshold=roi_threshold,
                vlm_model_resolved=vlm_model_resolved,
                vlm_crop_cache_size=vlm_crop_cache_size,
                vlm_dead_after_lost=vlm_dead_after_lost,
                vlm_batch_size=b,
                warmup_seconds=float(args.warmup_seconds),
                measure_seconds=float(args.measure_seconds),
                measure_only_after_roi_lock=measure_after_roi,
                vlm_state=shared_vlm,
            )
            extra_runs.append({"category": "extra_sweep", "vlm_batch_size": b, **st})

    payload = {
        "created_at": ts,
        "sample_videos": [{"label": a, "path": b} for a, b in resolved],
        "warmup_seconds": args.warmup_seconds,
        "measure_seconds": args.measure_seconds,
        "measure_only_after_roi_lock": measure_after_roi,
        "microbatch_size": mb,
        "cache_drain_batch": int(args.cache_drain_batch),
        "cache_drain_tasks": int(args.cache_drain_tasks),
        "per_sample": per_sample,
        "cache_flow": stats_c,
        "plot_matrices": {
            "trucks_per_sec_wall": trucks_mat,
            "vlm_queries_per_gpu_sec": gpu_mat,
            "panel3": panel3_mat,
        },
        "extra_sweep_runs": extra_runs,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    mode_titles = [
        "Baseline\n(sync B=1)",
        f"Microbatch\n(sync B={mb})",
        f"Cache flow\n(JSONL B={int(args.cache_drain_batch)})",
    ]
    title = (
        f"VLM modes × samples | device={device}\n"
        + ", ".join(sample_labels)
    )
    _plot_grouped(
        sample_labels=sample_labels,
        mode_titles=mode_titles,
        trucks=trucks_mat,
        gpu_qps=gpu_mat,
        panel3=panel3_mat,
        output_png=out_png,
        title=title,
    )

    print(f"[measure_vlm_modes] Wrote {out_png}")
    print(f"[measure_vlm_modes] Wrote {out_json}")


if __name__ == "__main__":
    main()
