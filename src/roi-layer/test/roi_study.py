#!/usr/bin/env python3
"""
roi_study.py

Empirically tests whether ROI cropping speeds up YOLO on this machine.

It does NOT run the full pipeline. It focuses on Layer 3 (ROI) + Layer 4 (YOLO):
- Run YOLO on full frames to drive ROI discovery until locked (same as pipeline behavior).
- After ROI locks, measure YOLO inference time on:
  - full frame
  - ROI-cropped frame

This isolates the "ROI helps YOLO?" question from VLM/tracking/etc.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"

CONFIG_DIR = SRC_DIR / "configuration-layer"
ROI_DIR = SRC_DIR / "roi-layer"
YOLO_DIR = SRC_DIR / "yolo-layer"
INPUT_DIR = SRC_DIR / "input-layer"

for p in (CONFIG_DIR, ROI_DIR, YOLO_DIR, INPUT_DIR):
    sys.path.insert(0, str(p))


def _ms(values_s: list[float]) -> float:
    if not values_s:
        return 0.0
    return statistics.mean(values_s) * 1000.0


def _p50_ms(values_s: list[float]) -> float:
    if not values_s:
        return 0.0
    return statistics.median(values_s) * 1000.0


def run_one_video(
    video_path: str,
    *,
    frame_resolution: tuple[int, int] = (640, 360),
    device: str = "cuda",
    yolo_model: str = "yolov10n.pt",
    yolo_conf: float = 0.45,
    roi_threshold: int = 10,
    warmup_frames: int = 15,
    measure_frames: int = 120,
) -> dict:
    import cv2

    from roi_layer import build_roi_layer_package, initialize_roi_layer, update_roi_state
    from detector import filter_yolo_detections, initialize_yolo_layer, run_yolo_detection

    initialize_roi_layer(config_roi_enabled=True, config_roi_vehicle_count_threshold=roi_threshold)
    initialize_yolo_layer(model_name=yolo_model, conf_threshold=yolo_conf, device=device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")

    def next_frame() -> tuple[int, object] | None:
        ok, frame = cap.read()
        if not ok:
            return None
        w, h = frame_resolution
        frame = cv2.resize(frame, (w, h))
        fid = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        return fid, frame

    roi_locked_at = None
    roi_bounds = None
    roi_area_frac = None

    # Phase 1: lock ROI (using YOLO detections on full frames, like pipeline).
    t0 = time.perf_counter()
    while True:
        nxt = next_frame()
        if nxt is None:
            break
        fid, frame = nxt
        input_pkg = {
            "input_layer_frame_id": fid,
            "input_layer_timestamp": time.time(),
            "input_layer_image": frame,
            "input_layer_source_type": "video",
            "input_layer_resolution": frame_resolution,
        }
        roi_pkg = build_roi_layer_package(input_pkg)
        dets_boot = run_yolo_detection(roi_pkg if roi_pkg.get("roi_layer_locked") else input_pkg)
        dets_boot = filter_yolo_detections(dets_boot)
        roi_state = update_roi_state(input_pkg, dets_boot)
        if roi_state.get("roi_layer_locked"):
            roi_locked_at = fid
            roi_bounds = tuple(int(v) for v in roi_state.get("roi_layer_bounds"))
            x1, y1, x2, y2 = roi_bounds
            roi_area_frac = ((x2 - x1) * (y2 - y1)) / float(frame_resolution[0] * frame_resolution[1])
            break
        # Guard: avoid scanning entire video if never locks.
        if (time.perf_counter() - t0) > 15.0:
            break

    if roi_bounds is None:
        cap.release()
        return {
            "video": video_path,
            "roi_locked": False,
            "roi_locked_at": None,
            "roi_bounds": None,
            "roi_area_frac": None,
            "note": "ROI did not lock quickly enough to measure post-ROI timing.",
        }

    # Phase 2: warmup + measure YOLO timings full vs ROI.
    full_times: list[float] = []
    roi_times: list[float] = []
    full_pixels: list[int] = []
    roi_pixels: list[int] = []

    # Warmup
    for _ in range(warmup_frames):
        nxt = next_frame()
        if nxt is None:
            break
        fid, frame = nxt
        input_pkg = {
            "input_layer_frame_id": fid,
            "input_layer_timestamp": time.time(),
            "input_layer_image": frame,
            "input_layer_source_type": "video",
            "input_layer_resolution": frame_resolution,
        }
        roi_pkg = build_roi_layer_package(input_pkg)
        _ = run_yolo_detection(input_pkg)
        _ = run_yolo_detection(roi_pkg)

    # Measure
    for _ in range(measure_frames):
        nxt = next_frame()
        if nxt is None:
            break
        fid, frame = nxt
        input_pkg = {
            "input_layer_frame_id": fid,
            "input_layer_timestamp": time.time(),
            "input_layer_image": frame,
            "input_layer_source_type": "video",
            "input_layer_resolution": frame_resolution,
        }
        roi_pkg = build_roi_layer_package(input_pkg)
        roi_img = roi_pkg["roi_layer_image"]

        full_pixels.append(int(frame.shape[0]) * int(frame.shape[1]))
        roi_pixels.append(int(roi_img.shape[0]) * int(roi_img.shape[1]))

        a0 = time.perf_counter()
        _ = run_yolo_detection(input_pkg)
        a1 = time.perf_counter()
        full_times.append(a1 - a0)

        b0 = time.perf_counter()
        _ = run_yolo_detection(roi_pkg)
        b1 = time.perf_counter()
        roi_times.append(b1 - b0)

    cap.release()
    return {
        "video": video_path,
        "roi_locked": True,
        "roi_locked_at": roi_locked_at,
        "roi_bounds": roi_bounds,
        "roi_area_frac": roi_area_frac,
        "yolo_full_mean_ms": _ms(full_times),
        "yolo_full_p50_ms": _p50_ms(full_times),
        "yolo_roi_mean_ms": _ms(roi_times),
        "yolo_roi_p50_ms": _p50_ms(roi_times),
        "avg_full_pixels": int(statistics.mean(full_pixels)) if full_pixels else 0,
        "avg_roi_pixels": int(statistics.mean(roi_pixels)) if roi_pixels else 0,
        "roi_speedup_factor": (statistics.mean(full_times) / statistics.mean(roi_times)) if full_times and roi_times else None,
    }


def main() -> None:
    videos = [
        str(REPO_ROOT / "data" / f"sample{i}.mp4")
        for i in (1, 2, 3, 4)
    ]
    for video in videos:
        if not Path(video).is_file():
            continue
        result = run_one_video(video)
        print()
        print(video)
        for k, v in result.items():
            if k == "video":
                continue
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()

