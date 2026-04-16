#!/usr/bin/env python3
"""
benchmark.py (repo root)

Runs a short, end-to-end *video-only* pipeline pass and prints timing + throughput
to help assess device usability for this stack.

It reads `src/configuration-layer/config.yaml` and exercises:
input(video) -> ROI (if enabled) -> YOLO -> tracking -> vehicle-state
optionally scene-awareness (if enabled), and optionally VLM inference (if enabled).

When `config_vlm_enabled` is true and PyTorch + transformers are available, this
script runs the cropper→dispatch→VLM path using `config_vlm_runtime_mode`
(inline = sync in the frame loop; async/spill = background worker like the realtime visualizer).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# On Jetson unified-memory (NvMap) devices the CUDACachingAllocator can hit
# internal assertions when it tries to pre-allocate or cache large blocks.
# Limiting the split size to 128 MB keeps individual NvMap allocations small
# enough to succeed, while still allowing the full model weight set to be
# loaded as many smaller chunks.  Must be set before any CUDA init.
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Enable cuDNN auto-tuner so it selects the fastest convolution algorithm for
# each fixed input shape encountered during the run. This is a net win for
# the .pt (eager PyTorch) YOLO path and is a no-op for TensorRT engines.
try:
    import torch
    torch.backends.cudnn.benchmark = True
except Exception:
    pass


REPO_ROOT = Path(__file__).resolve().parent
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

# Optional override for the VLM micro-batch size used by this benchmark.
# When None, defaults to `config_vlm_worker_batch_size` from config.yaml.
BENCH_OVERRIDE_VLM_BATCH_SIZE: int | None = None

# Fallback (frame-count mode). Used only if MEASURE_SECONDS <= 0.
WARMUP_FRAMES = 3
MEASURED_FRAMES = 30

# Allow overriding the config file path via environment variable so different
# Jetson-optimized configs can be tested without editing this file.
# Example:
#   BENCH_CONFIG_YAML=/app/src/configuration-layer/config.jetson.yaml python3 benchmark.py
_config_yaml_override = os.environ.get("BENCH_CONFIG_YAML", "").strip()

CONFIG_DIR = SRC_DIR / "configuration-layer"
INPUT_DIR = SRC_DIR / "input-layer"
ROI_DIR = SRC_DIR / "roi-layer"
YOLO_DIR = SRC_DIR / "yolo-layer"
TRACKING_DIR = SRC_DIR / "tracking-layer"
VSTATE_DIR = SRC_DIR / "vehicle-state-layer"
SCENE_DIR = SRC_DIR / "scene-awareness-layer"
VLM_DIR = SRC_DIR / "vlm-layer"
VLM_UTIL_DIR = VLM_DIR / "util"
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


def _p(label: str, value: str) -> None:
    print(f"[profile] {label}: {value}")


def _safe_div(n: float, d: float) -> float:
    return n / d if d not in (0.0, -0.0) else 0.0


def _fmt_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _bench_mark(tier: str, col: str) -> str:
    return "  x  " if tier == col else "     "


def _print_health_table(rows: list[tuple[str, str, str]]) -> None:
    """Print good/meh/problem columns. Each row: (label, tier, detail)."""
    lw = 32
    print()
    print("Health (snapshot of this run; not a guarantee under all traffic)")
    print(
        "  Note: 'FPS vs source' is only about live frame sync; "
        "if your goal is eventual truck logging (not real-time preview), a slower pipeline can still be OK."
    )
    header = f"{'check':{lw}} | good | meh | problem | detail"
    print(header)
    print("-" * min(120, len(header) + 48))
    for label, tier, detail in rows:
        t = tier if tier in ("good", "meh", "problem") else "meh"
        print(
            f"{label:{lw}} |{_bench_mark(t, 'good')}|{_bench_mark(t, 'meh')}|{_bench_mark(t, 'problem')}| {detail}"
        )
    print()


def _tier_fps_vs_source(src_fps: float, fps: float) -> tuple[str, str]:
    """Tier live throughput vs video FPS. Logging-only workloads do not need to match source FPS."""
    if src_fps <= 0:
        return "meh", "source FPS unknown — cannot compare"
    if fps <= 0:
        return "problem", "pipeline produced ~0 fps — no forward progress"
    if fps - src_fps >= -0.5:
        return "good", f"pipeline {fps:.1f} fps ~ source {src_fps:.1f} fps (live sync OK)"
    if fps >= src_fps * 0.85:
        return "meh", (
            f"pipeline {fps:.1f} fps slightly under source {src_fps:.1f} fps — "
            "live preview would lag; still OK for eventual logging if tracks get processed"
        )
    return "meh", (
        f"pipeline {fps:.1f} fps under source {src_fps:.1f} fps — "
        "not real-time vs camera; fine if you only need trucks logged over time, not per-frame sync"
    )


def _tier_yolo(src_fps: float, yolo_cap_fps: float) -> tuple[str, str]:
    if src_fps <= 0 or yolo_cap_fps <= 0:
        return "meh", "YOLO capacity vs source not assessed"
    if yolo_cap_fps + 0.5 >= src_fps:
        return "good", f"YOLO throughput {yolo_cap_fps:.1f} fps ≥ source {src_fps:.1f} fps"
    if yolo_cap_fps >= src_fps * 0.85:
        return "meh", f"YOLO {yolo_cap_fps:.1f} fps close to source {src_fps:.1f} fps"
    return "problem", f"YOLO {yolo_cap_fps:.1f} fps below source {src_fps:.1f} fps"


def _tier_roi(
    roi_enabled: bool,
    roi_lock_frame_id: int | None,
    roi_frac: float | None,
) -> tuple[str, str]:
    if not roi_enabled:
        return "meh", "ROI disabled (full-frame YOLO)"
    if roi_lock_frame_id is None:
        return "problem", "ROI enabled but not locked during measured window"
    if roi_frac is not None and roi_frac >= 0.9:
        return "meh", f"ROI locked but covers {_fmt_pct(roi_frac)} of frame — limited YOLO win"
    return "good", "ROI locked with smaller-than-frame crop"


def _tier_vlm(
    vlm_enabled: bool,
    vlm_available: bool,
    vlm_query_count: int,
    calls_per_s: float,
    avg_query_ms: float,
    target_ms: float,
    spilled: int,
) -> tuple[str, str]:
    if not vlm_enabled:
        return "good", "VLM off in config"
    if not vlm_available:
        return "problem", "VLM enabled but failed to load / skipped"
    if vlm_query_count <= 0 or calls_per_s <= 0:
        return "meh", "no VLM queries completed — stability not tested"
    eps = 0.5
    if spilled > 0:
        return "problem", f"spill occurred ({spilled}x) — worker queue overflowed"
    if avg_query_ms <= target_ms - eps:
        return "good", f"avg {avg_query_ms:.0f} ms/query ≤ budget {target_ms:.0f} ms"
    if avg_query_ms <= target_ms + eps:
        return "meh", f"avg {avg_query_ms:.0f} ms/query ~ budget {target_ms:.0f} ms (marginal)"
    return "problem", f"avg {avg_query_ms:.0f} ms/query > budget {target_ms:.0f} ms"


def _emit_profile_blocks(
    *,
    frames_done: int,
    det_counts: int,
    track_counts: int,
    vlm_calls: int,
    e2e_ms: float,
    fps: float,
    totals: dict[str, float],
    roi_enabled: bool,
    roi_waited_s: float,
    roi_lock_frame_id: int | None,
    vlm_enabled: bool,
    vlm_available: bool,
    vlm_effective_runtime: str,
    vlm_drain_elapsed_s: float,
    vlm_metrics_elapsed_s: float,
    measure_elapsed_s: float,
) -> None:
    print("[profile] --- run ---")
    _p("frames_processed", str(frames_done))
    _p("avg_detections_per_frame", f"{det_counts / frames_done:.2f}")
    _p("avg_tracks_per_frame", f"{track_counts / frames_done:.2f}")
    _p("avg_vlm_calls_per_frame", f"{vlm_calls / frames_done:.3f}")
    _p("avg_end_to_end_ms", f"{e2e_ms:.2f}")
    _p("estimated_fps", f"{fps:.2f}")
    print("[profile] --- per-frame avg ms (measured window) ---")
    _p("avg_input_ms", f"{(totals['input_s'] / frames_done) * 1000.0:.2f}")
    _p("avg_roi_ms", f"{(totals['roi_s'] / frames_done) * 1000.0:.2f}")
    _p("avg_yolo_ms", f"{(totals['yolo_s'] / frames_done) * 1000.0:.2f}")
    _p("avg_tracking_ms", f"{(totals['tracking_s'] / frames_done) * 1000.0:.2f}")
    _p("avg_state_ms", f"{(totals['state_s'] / frames_done) * 1000.0:.2f}")
    _p("avg_vlm_ms", f"{(totals['vlm_s'] / frames_done) * 1000.0:.2f} (main thread: submit+drain when async)")
    _p("avg_scene_ms", f"{(totals['scene_s'] / frames_done) * 1000.0:.2f}")
    if roi_enabled and MEASURE_ONLY_AFTER_ROI_LOCK:
        print("[profile] --- roi benchmark gate ---")
        _p("roi_lock_waited_s", f"{roi_waited_s:.2f}")
        _p("roi_locked_before_measure", str(roi_lock_frame_id is not None))
    if vlm_enabled and vlm_available and vlm_effective_runtime in ("async", "spill"):
        print("[profile] --- vlm async window ---")
        _p("measure_window_s", f"{measure_elapsed_s:.2f}")
        _p("vlm_post_measure_drain_s", f"{vlm_drain_elapsed_s:.2f}")
        _p("vlm_metrics_elapsed_s", f"{vlm_metrics_elapsed_s:.2f}")


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

    config_path = Path(_config_yaml_override) if _config_yaml_override else CONFIG_DIR / "config.yaml"
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
    _yolo_imgsz_raw = get_config_value(cfg, "config_yolo_imgsz")
    yolo_imgsz = tuple(int(x) for x in _yolo_imgsz_raw) if _yolo_imgsz_raw else None
    roi_enabled = bool(get_config_value(cfg, "config_roi_enabled"))
    if BENCH_OVERRIDE_ROI_ENABLED is not None:
        roi_enabled = bool(BENCH_OVERRIDE_ROI_ENABLED)
    roi_threshold = int(get_config_value(cfg, "config_roi_vehicle_count_threshold"))
    scene_enabled = bool(get_config_value(cfg, "config_scene_awareness_enabled"))

    vlm_enabled = bool(get_config_value(cfg, "config_vlm_enabled"))
    vlm_model = str(get_config_value(cfg, "config_vlm_model"))
    # config_vlm_device overrides config_device for the VLM only.
    # Empty string means inherit from config_device (default behaviour).
    _vlm_device_override = str(get_config_value(cfg, "config_vlm_device") or "").strip()
    vlm_device = _vlm_device_override if _vlm_device_override else device
    vlm_crop_cache_size = int(get_config_value(cfg, "config_vlm_crop_cache_size"))
    vlm_dead_after_lost_frames = int(get_config_value(cfg, "config_vlm_dead_after_lost_frames"))
    vlm_worker_batch_wait_ms = int(get_config_value(cfg, "config_vlm_worker_batch_wait_ms"))
    vlm_worker_max_queue_size = int(get_config_value(cfg, "config_vlm_worker_max_queue_size"))
    vlm_runtime_mode_cfg = str(get_config_value(cfg, "config_vlm_runtime_mode"))
    vlm_bench_batch_size = int(get_config_value(cfg, "config_vlm_worker_batch_size"))
    if BENCH_OVERRIDE_VLM_BATCH_SIZE is not None:
        vlm_bench_batch_size = int(BENCH_OVERRIDE_VLM_BATCH_SIZE)
    vlm_bench_batch_size = max(1, int(vlm_bench_batch_size))
    vlm_bench_mode = str(vlm_runtime_mode_cfg or "inline").strip().lower()
    if vlm_bench_mode not in ("inline", "async", "spill"):
        vlm_bench_mode = "inline"
    vlm_runtime_mode_intent = vlm_bench_mode

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
    if vlm_enabled:
        _p("vlm_device", vlm_device)
        _p("vlm_benchmark_batch_size", str(int(vlm_bench_batch_size)))
        _p("vlm_benchmark_runtime_mode_intent", vlm_runtime_mode_intent)
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
    initialize_yolo_layer(model_name=model, conf_threshold=conf, device=device, imgsz=yolo_imgsz)
    initialize_tracking_layer(frame_rate=30)
    initialize_vehicle_state_layer(prune_after_lost_frames=None)

    # Initialize crop cache + VLM runtime if enabled and deps exist.
    vlm_available = False
    vlm_state = None
    crop_cache = None
    vlm_worker = None
    vlm_spill_path_resolved = ""
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
                prepare_vlm_prompt,
                run_vlm_inference_batch,
            )

            vlm_state = initialize_vlm_layer(
                VLMConfig(
                    config_vlm_enabled=True,
                    config_vlm_model=_resolve_repo_path(vlm_model),
                    config_device=vlm_device,
                )
            )
            vlm_runtime_device = str(getattr(vlm_state, "vlm_runtime_device", "unknown"))
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
                            vlm_spill_path_resolved = spill_path
                        spill_mb = float(get_config_value(cfg, "config_vlm_spill_max_file_mb"))
                        spill_max_bytes = int(spill_mb * 1024 * 1024) if spill_mb > 0 else 0
                    vlm_crop_feedback_enabled = bool(get_config_value(cfg, "config_vlm_crop_feedback_enabled"))
                    vlm_worker = AsyncVLMWorker(
                        vlm_state=vlm_state,
                        feedback_enabled=vlm_crop_feedback_enabled,
                        max_queue_size=vlm_worker_max_queue_size,
                        batch_size=vlm_bench_batch_size,
                        batch_wait_ms=vlm_worker_batch_wait_ms,
                        spill_queue_path=spill_path,
                        spill_max_file_bytes=spill_max_bytes,
                    )
                    vlm_worker.start()
                except Exception as exc:
                    _p("vlm_worker_init", f"failed; using inline VLM ({type(exc).__name__}: {exc})")
                    vlm_worker = None
        except Exception as exc:
            _p("vlm_status", f"SKIP ({type(exc).__name__}: {exc})")
            vlm_available = False

    vlm_effective_runtime = vlm_runtime_mode_intent
    if vlm_enabled and vlm_available and vlm_runtime_mode_intent in ("async", "spill") and vlm_worker is None:
        vlm_effective_runtime = "inline"
    if vlm_enabled and vlm_available:
        _p("vlm_benchmark_runtime_effective", vlm_effective_runtime)

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

    def _vlm_flush_worker_queue_for_handoff() -> None:
        if vlm_worker is None or crop_cache is None:
            return
        deadline = time.perf_counter() + 300.0
        while time.perf_counter() < deadline:
            for result in vlm_worker.drain_results():
                register_vlm_ack_package(crop_cache, result["ack"])
            st = vlm_worker.get_status()
            if st["queue_size"] == 0 and not st["busy"]:
                break
            time.sleep(0.02)

    def _vlm_drain_worker_results() -> None:
        nonlocal vlm_query_s_total, vlm_query_count, vlm_calls
        if vlm_worker is None or crop_cache is None:
            return
        batch_results = vlm_worker.drain_results()
        if not batch_results:
            return
        i = 0
        n = len(batch_results)
        while i < n:
            rt = float(batch_results[i]["runtime_sec"])
            j = i + 1
            while j < n and float(batch_results[j]["runtime_sec"]) == rt:
                j += 1
            wall = rt
            nbatch = j - i
            vlm_query_s_total += wall
            vlm_query_count += nbatch
            vlm_calls += nbatch
            for k in range(i, j):
                register_vlm_ack_package(crop_cache, batch_results[k]["ack"])
            i = j

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

        # Single YOLO call per frame: result feeds both the pipeline and ROI state.
        # Before ROI lock: runs on the full frame (same input update_roi_state needs).
        # After ROI lock: update_roi_state is a no-op (returns immediately), so the
        # previous pattern of a separate dets_boot YOLO call was entirely wasted.
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

        # Update ROI state with this frame's detections for next frame's ROI crop.
        roi_state = update_roi_state(input_pkg, dets)
        roi_last_candidate_count = int(roi_state.get("roi_candidate_box_count", 0))
        roi_last_locked = bool(roi_state.get("roi_layer_locked", False))
        bounds = roi_state.get("roi_layer_bounds")
        roi_last_bounds = tuple(int(v) for v in bounds) if bounds is not None else None
        if roi_lock_frame_id is None and roi_last_locked:
            roi_lock_frame_id = int(input_pkg["input_layer_frame_id"])
            roi_lock_candidate_count = roi_last_candidate_count
            if roi_last_bounds is not None:
                roi_lock_bounds = roi_last_bounds
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

            if vlm_effective_runtime in ("async", "spill") and vlm_worker is not None:
                _vlm_drain_worker_results()
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
                dispatch_pkgs: list[VLMFrameCropperLayerPackage] = []
                batch_size = int(vlm_bench_batch_size)
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
                    if len(dispatch_pkgs) >= batch_size:
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

    _vlm_flush_worker_queue_for_handoff()

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

    vlm_spilled_final = 0
    vlm_drain_elapsed_s = 0.0
    t_vlm_drain0 = time.perf_counter()
    if vlm_worker is not None:
        deadline = time.perf_counter() + 300.0
        while time.perf_counter() < deadline:
            _vlm_drain_worker_results()
            st = vlm_worker.get_status()
            if st["queue_size"] == 0 and not st["busy"]:
                break
            time.sleep(0.02)
        _vlm_drain_worker_results()
        vlm_spilled_final = int(getattr(vlm_worker, "spilled_count", 0) or 0)
        vlm_worker.shutdown()
    vlm_drain_elapsed_s = max(0.0, time.perf_counter() - t_vlm_drain0)

    vlm_metrics_elapsed_s = (
        measure_elapsed_s + vlm_drain_elapsed_s
        if (vlm_enabled and vlm_available and vlm_effective_runtime in ("async", "spill"))
        else measure_elapsed_s
    )

    if frames_done <= 0:
        raise RuntimeError("No frames processed in measured section.")

    e2e_ms = (totals["end_to_end_s"] / frames_done) * 1000.0
    fps = 1.0 / max(1e-9, totals["end_to_end_s"] / frames_done)
    _emit_profile_blocks(
        frames_done=frames_done,
        det_counts=det_counts,
        track_counts=track_counts,
        vlm_calls=vlm_calls,
        e2e_ms=e2e_ms,
        fps=fps,
        totals=totals,
        roi_enabled=roi_enabled,
        roi_waited_s=roi_waited_s,
        roi_lock_frame_id=roi_lock_frame_id,
        vlm_enabled=vlm_enabled,
        vlm_available=vlm_available,
        vlm_effective_runtime=vlm_effective_runtime,
        vlm_drain_elapsed_s=vlm_drain_elapsed_s,
        vlm_metrics_elapsed_s=vlm_metrics_elapsed_s,
        measure_elapsed_s=measure_elapsed_s,
    )

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
    yolo_cap_fps = _safe_div(float(frames_done), totals["yolo_s"]) if totals["yolo_s"] > 0 else 0.0
    roi_frac: float | None = None
    if roi_lock_bounds is not None:
        x1, y1, x2, y2 = roi_lock_bounds
        roi_area = max(1, (x2 - x1) * (y2 - y1))
        full_area = max(1, int(frame_resolution[0]) * int(frame_resolution[1]))
        roi_frac = roi_area / float(full_area)

    calls_per_s = float(vlm_calls) / float(vlm_metrics_elapsed_s) if vlm_metrics_elapsed_s > 0 else 0.0
    target_ms = (1.0 / calls_per_s) * 1000.0 if calls_per_s > 0 else 0.0

    t_in, t_roi, t_yolo, t_vlm = (
        _tier_fps_vs_source(src_fps, fps),
        _tier_roi(roi_enabled, roi_lock_frame_id, roi_frac),
        _tier_yolo(src_fps, yolo_cap_fps),
        _tier_vlm(
            vlm_enabled,
            vlm_available,
            vlm_query_count,
            calls_per_s,
            avg_vlm_query_ms,
            target_ms,
            vlm_spilled_final if vlm_effective_runtime == "spill" else 0,
        ),
    )
    health_rows = [
        ("FPS vs source (live sync only)", t_in[0], t_in[1]),
        ("ROI (lock & crop usefulness)", t_roi[0], t_roi[1]),
        ("YOLO throughput vs source", t_yolo[0], t_yolo[1]),
        ("VLM keep-up / queue", t_vlm[0], t_vlm[1]),
    ]
    _print_health_table(health_rows)

    print("Details (same data as before, grouped tighter)")
    print()
    print("Input")
    print(f"  source_fps={src_fps:.2f}" if src_fps > 0 else "  source_fps=unknown")
    print(f"  measured_pipeline_fps={fps:.2f}")
    print()

    print("ROI (layer 3)")
    if roi_enabled:
        if roi_lock_frame_id is None:
            print(f"  not locked in window | last_candidates={roi_last_candidate_count}/{roi_threshold}")
            if roi_last_bounds is not None:
                print(f"  last_bounds={roi_last_bounds}")
        else:
            print(f"  locked frame={roi_lock_frame_id} candidates={roi_lock_candidate_count}")
            if roi_lock_bounds is not None and roi_frac is not None:
                print(f"  bounds={roi_lock_bounds} | area={_fmt_pct(roi_frac)} of frame")
    else:
        print("  disabled")
    print()

    print("YOLO (layer 4)")
    if yolo_full_frames > 0:
        print(
            f"  pre-ROI n={yolo_full_frames} avg_ms={avg_yolo_full_s * 1000.0:.2f} "
            f"cap_fps={yolo_full_capacity_fps:.1f} infer/post="
            f"{avg_yolo_full_infer_s * 1000.0:.1f}/{avg_yolo_full_post_s * 1000.0:.1f} "
            f"px={int(_safe_div(float(yolo_full_pixels_total), float(yolo_full_frames))):,} "
            f"dets={_safe_div(float(yolo_full_det_total), float(yolo_full_frames)):.2f}/f"
        )
    if yolo_roi_frames > 0:
        print(
            f"  post-ROI n={yolo_roi_frames} avg_ms={avg_yolo_roi_s * 1000.0:.2f} "
            f"cap_fps={yolo_roi_capacity_fps:.1f} infer/post="
            f"{avg_yolo_roi_infer_s * 1000.0:.1f}/{avg_yolo_roi_post_s * 1000.0:.1f} "
            f"px={int(_safe_div(float(yolo_roi_pixels_total), float(yolo_roi_frames))):,} "
            f"dets={_safe_div(float(yolo_roi_det_total), float(yolo_roi_frames)):.2f}/f"
        )
        if avg_yolo_full_s > 0 and avg_yolo_roi_s > 0:
            print(f"  ROI YOLO speedup {avg_yolo_full_s / avg_yolo_roi_s:.2f}x")
    print(f"  end-to-end YOLO capacity ~{yolo_cap_fps:.2f} fps (total_yolo_s / frames)")
    print()

    print("VLM (layer 8)")
    if not vlm_enabled:
        print("  disabled")
    elif not vlm_available:
        print("  skipped (deps / load failure)")
    else:
        fb = " (worker init failed → inline)" if vlm_runtime_mode_intent in ("async", "spill") and vlm_effective_runtime == "inline" else ""
        print(f"  device={vlm_runtime_device} (config_device={device})")
        print(
            f"  config_mode={vlm_runtime_mode_cfg} ran_as={vlm_effective_runtime}{fb} | "
            f"batch≤{vlm_bench_batch_size} wait_ms={vlm_worker_batch_wait_ms} | "
            f"cache={vlm_crop_cache_size} queue={vlm_worker_max_queue_size}"
        )
        if vlm_effective_runtime == "spill" and vlm_spill_path_resolved:
            print(f"  spill_path={vlm_spill_path_resolved}")
        if vlm_effective_runtime == "spill" and vlm_spilled_final > 0:
            print(f"  spill_events={vlm_spilled_final}")
        if vlm_query_count > 0:
            qkind = "GPU worker amortized" if vlm_effective_runtime in ("async", "spill") else "sync in frame loop"
            print(f"  avg_query_ms={avg_vlm_query_ms:.1f} ({qkind}) | queries={vlm_query_count}")
        else:
            print("  no queries")
        print(f"  calls_per_frame={vlm_calls / frames_done:.3f}")
        if calls_per_s > 0:
            window_note = (
                f"measure+drain={vlm_metrics_elapsed_s:.2f}s"
                if vlm_effective_runtime in ("async", "spill")
                else f"measure={measure_elapsed_s:.2f}s"
            )
            print(f"  completion_rate={calls_per_s:.3f}/s ({window_note})")
            print(f"  budget_avg_query_ms≤{target_ms:.0f} (1 worker, steady traffic)")
            if vlm_query_count > 0 and avg_vlm_query_ms > 0:
                print(f"  implied_service_qps={1000.0 / avg_vlm_query_ms:.3f}")
            eps_ms = 0.5
            if avg_vlm_query_ms <= target_ms - eps_ms:
                tail = (
                    f"worker wait={vlm_worker_batch_wait_ms}ms queue_cap={vlm_worker_max_queue_size}"
                    if vlm_effective_runtime in ("async", "spill")
                    else "inline: batch_wait_ms N/A"
                )
                print(f"  outcome: likely keeps up ({tail})")
            elif avg_vlm_query_ms <= target_ms + eps_ms:
                print("  outcome: marginal vs budget")
            else:
                service_s = avg_vlm_query_ms / 1000.0
                mu = 1.0 / max(1e-12, service_s)
                drift = calls_per_s - mu
                if drift <= 1e-9:
                    print("  outcome: marginal ~ service rate matches completion rate")
                else:
                    qcap = max(1, int(vlm_worker_max_queue_size))
                    sec_to_fill = qcap / drift
                    if sec_to_fill >= 3600.0:
                        t_human = f"{sec_to_fill / 3600.0:.2f} h"
                    elif sec_to_fill >= 60.0:
                        t_human = f"{sec_to_fill / 60.0:.2f} min"
                    else:
                        t_human = f"{sec_to_fill:.1f} s"
                    if vlm_effective_runtime in ("async", "spill"):
                        print(f"  outcome: backlog ~{drift:.3f} q/s; queue {qcap} full in ~{t_human}")
                    else:
                        print(
                            f"  outcome: behind ~{drift:.3f} q/s vs one worker; "
                            f"hypothetical queue {qcap} full in ~{t_human} (inline, no worker queue)"
                        )
        else:
            print("  outcome: no VLM completions in metrics window — skip keep-up math")


def _git_branch() -> str:
    try:
        import subprocess

        out = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO_ROOT))
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()

