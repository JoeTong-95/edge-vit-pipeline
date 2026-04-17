#!/usr/bin/env python3
"""
capture_video.py

Capture live camera footage and save to the data/ folder.

Usage:
    python capture_video.py
    python capture_video.py --device 0 --width 1280 --height 720 --fps 30
    python capture_video.py --use-gstreamer   # Jetson CSI camera

Steps:
    1. Opens the camera and probes actual frame size/FPS.
    2. Calculates free disk space and shows the maximum recordable duration.
    3. Prompts for a desired recording duration (seconds).
    4. Records and saves to data/<timestamp>.mp4
    5. Prints a summary when done.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src" / "input-layer"))

from input_layer import InputLayer  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
DATA_DIR = _ROOT / "data"
# Reserve this much disk so the OS stays healthy
RESERVED_DISK_BYTES = 512 * 1024 * 1024  # 512 MB
# Safety factor: assume encoded video is ~80 % of raw size
ENCODE_RATIO = 0.80


# ── helpers ───────────────────────────────────────────────────────────────────

def _free_bytes() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(DATA_DIR)
    return max(0, free - RESERVED_DISK_BYTES)


def _raw_bytes_per_second(width: int, height: int, fps: float) -> float:
    """BGR bytes / second before encoding."""
    return width * height * 3 * fps


def _max_record_seconds(width: int, height: int, fps: float) -> float:
    usable = _free_bytes() * (1.0 / ENCODE_RATIO)
    raw_bps = _raw_bytes_per_second(width, height, fps)
    if raw_bps <= 0:
        return 0.0
    return usable / raw_bps


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _fmt_bytes(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:.1f} {unit}"
        b /= 1024
    return str(b)


def _ask_duration(max_seconds: float) -> float:
    while True:
        raw = input(f"  Enter desired recording length in seconds (max {int(max_seconds)}s): ").strip()
        try:
            val = float(raw)
        except ValueError:
            print("  ✗ Please enter a number.")
            continue
        if val <= 0:
            print("  ✗ Duration must be positive.")
        elif val > max_seconds:
            print(f"  ✗ That exceeds the maximum ({int(max_seconds)}s). Try a shorter value.")
        else:
            return val


def _probe_camera(
    device: int,
    use_gstreamer: bool,
    width: int,
    height: int,
    fps: int,
) -> tuple[int, int, float]:
    """Open camera briefly to confirm actual resolution and FPS."""
    if use_gstreamer:
        pipeline = (
            f"nvarguscamerasrc sensor-id={device} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"format=NV12, framerate={fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={width}, height={height}, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        print(f"  ✗ Could not open camera device {device}.")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
    cap.release()
    return actual_w, actual_h, actual_fps


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture live camera footage and save to data/"
    )
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Capture height (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--use-gstreamer", action="store_true",
                        help="Use GStreamer pipeline for Jetson CSI camera")
    args = parser.parse_args()

    print()
    print("═" * 60)
    print("  EDGE-VIT PIPELINE — VIDEO CAPTURE")
    print("═" * 60)

    # ── probe camera ──────────────────────────────────────────────
    print(f"\n[1/4] Probing camera (device={args.device}) …")
    actual_w, actual_h, actual_fps = _probe_camera(
        args.device, args.use_gstreamer, args.width, args.height, args.fps
    )
    print(f"      Resolution : {actual_w} × {actual_h}")
    print(f"      FPS        : {actual_fps:.1f}")

    # ── disk space ────────────────────────────────────────────────
    print(f"\n[2/4] Calculating available disk space …")
    free = _free_bytes()
    max_secs = _max_record_seconds(actual_w, actual_h, actual_fps)
    print(f"      Usable space  : {_fmt_bytes(free)}")
    print(f"      Max record    : {_fmt_duration(max_secs)}  (at {actual_w}×{actual_h} @ {actual_fps:.0f}fps)")

    if max_secs < 5:
        print("\n  ✗ Not enough disk space to record (< 5 seconds usable). Free up space and retry.")
        sys.exit(1)

    # ── ask for duration ──────────────────────────────────────────
    print(f"\n[3/4] Recording duration")
    desired_secs = _ask_duration(max_secs)

    # ── output path ───────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = DATA_DIR / f"capture_{timestamp}.mp4"

    # ── record ────────────────────────────────────────────────────
    print(f"\n[4/4] Recording → {out_path}")
    print(f"      Press Ctrl+C to stop early.\n")

    input_layer = InputLayer()
    input_layer.initialize_input_layer(
        config_input_source="camera",
        config_frame_resolution=(actual_w, actual_h),
        camera_device_index=args.device,
        use_gstreamer=args.use_gstreamer,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, actual_fps, (actual_w, actual_h))
    if not writer.isOpened():
        print("  ✗ Failed to open VideoWriter. Check codec availability.")
        input_layer.close_input_layer()
        sys.exit(1)

    frames_written = 0
    t_start = time.monotonic()
    t_last_print = t_start

    try:
        while True:
            elapsed = time.monotonic() - t_start
            if elapsed >= desired_secs:
                break

            frame = input_layer.read_next_frame()
            if frame is None:
                print("  ⚠  Camera returned no frame — retrying …")
                time.sleep(0.05)
                continue

            writer.write(frame)
            frames_written += 1

            now = time.monotonic()
            if now - t_last_print >= 2.0:
                remaining = desired_secs - elapsed
                fps_actual = frames_written / max(elapsed, 0.001)
                print(f"  ● {elapsed:5.1f}s / {desired_secs:.0f}s  |  "
                      f"{frames_written} frames  |  {fps_actual:.1f} fps  |  "
                      f"{remaining:.0f}s remaining")
                t_last_print = now

    except KeyboardInterrupt:
        print("\n  ⚠  Stopped early by user.")

    finally:
        writer.release()
        input_layer.close_input_layer()

    elapsed_total = time.monotonic() - t_start
    file_size = out_path.stat().st_size if out_path.exists() else 0
    avg_fps = frames_written / max(elapsed_total, 0.001)

    print()
    print("═" * 60)
    print("  RECORDING COMPLETE")
    print("═" * 60)
    print(f"  File      : {out_path}")
    print(f"  Duration  : {_fmt_duration(elapsed_total)}")
    print(f"  Frames    : {frames_written}")
    print(f"  Avg FPS   : {avg_fps:.1f}")
    print(f"  File size : {_fmt_bytes(file_size)}")
    print()


if __name__ == "__main__":
    main()
