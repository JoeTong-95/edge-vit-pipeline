#!/usr/bin/env python3
"""
scheduled_capture.py

Automated camera capture at scheduled time windows.
Records raw video from a USB camera and saves MP4 files.

After recording, run the pipeline on saved files:
    python initialize_pipeline.py --video data/captures/morning_2026-04-22.mp4

Usage:
    python scheduled_capture.py                    # Run with default schedule
    python scheduled_capture.py --once             # Record one session immediately (for testing)
    python scheduled_capture.py --duration 60      # Override duration to 60 seconds (for testing)
    python scheduled_capture.py --list             # Show the schedule and exit

The script sleeps between windows and wakes up to record automatically.
Press Ctrl+C at any time to stop.

Output files are saved to data/captures/ with names like:
    morning_2026-04-22_07-00.mp4
    afternoon_2026-04-22_12-00.mp4
    evening_2026-04-22_17-00.mp4
    night_2026-04-22_21-00.mp4
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import cv2


# ===========================================================================
# SCHEDULE CONFIGURATION — EDIT THIS SECTION
# ===========================================================================
# Each entry: (label, start_hour, start_minute, duration_minutes)
#
# Examples:
#   ("morning", 7, 0, 60)    → Record 7:00 AM - 8:00 AM (60 min)
#   ("afternoon", 12, 0, 60) → Record 12:00 PM - 1:00 PM (60 min)
#   ("evening", 17, 0, 60)   → Record 5:00 PM - 6:00 PM (60 min)
#   ("night", 21, 0, 60)     → Record 9:00 PM - 10:00 PM (60 min)
#
# To change the schedule, just edit the times and durations below.
# To add more windows, add more tuples.
# To remove a window, delete or comment out the line.

CAPTURE_SCHEDULE = [
    ("morning",   7,  0,  120),
    ("afternoon", 13, 15,  120),
    ("evening",   16, 0,  120),
    ("night",     21, 0,  60),
    ("midnight",  0, 0, 60),
]

# ===========================================================================
# CAMERA CONFIGURATION — EDIT THIS SECTION
# ===========================================================================

CAMERA_DEVICE = 0           # /dev/video0
FRAME_WIDTH = 1280          # capture width
FRAME_HEIGHT = 720          # capture height
FPS = 30                    # target frames per second
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "captures")

# ===========================================================================
# VIDEO CODEC — usually no need to change
# ===========================================================================
# mp4v works everywhere; avc1/h264 is smaller but may need ffmpeg
FOURCC = "mp4v"


def get_next_window(now: datetime) -> tuple:
    """
    Find the next scheduled capture window from now.

    Returns:
        (label, start_datetime, duration_minutes) for the next window,
        or None if all today's windows have passed (wraps to tomorrow).
    """
    today = now.date()

    # Check remaining windows today
    for label, hour, minute, duration in CAPTURE_SCHEDULE:
        window_start = datetime(today.year, today.month, today.day, hour, minute)
        window_end = window_start + timedelta(minutes=duration)

        # If we're before the window ends, this is our next window
        if now < window_end:
            return (label, window_start, duration)

    # All windows passed today — wrap to first window tomorrow
    tomorrow = today + timedelta(days=1)
    label, hour, minute, duration = CAPTURE_SCHEDULE[0]
    window_start = datetime(tomorrow.year, tomorrow.month, tomorrow.day, hour, minute)
    return (label, window_start, duration)


def generate_filename(label: str, start_time: datetime) -> str:
    """Generate output filename, adding _2, _3 etc if file already exists."""
    date_str = start_time.strftime("%Y-%m-%d_%H-%M")
    base_path = os.path.join(OUTPUT_DIR, f"{label}_{date_str}")
    path = f"{base_path}.mp4"
    counter = 2
    while os.path.exists(path):
        path = f"{base_path}_{counter}.mp4"
        counter += 1
    return path


def record_session(label: str, duration_minutes: int, output_path: str) -> bool:
    """
    Record video from the camera for the specified duration.
    Uses ffmpeg subprocess for H.264 compression (much smaller files).
    Returns True if recording completed, False if it failed.
    """
    import subprocess

    print(f"\n{'=' * 60}")
    print(f"  RECORDING: {label}")
    print(f"  Duration:  {duration_minutes} minutes")
    print(f"  Output:    {output_path}")
    print(f"  Codec:     H.264 (libx264)")
    print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    # Open camera
    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        print(f"  ERROR: Could not open camera device {CAMERA_DEVICE}")
        return False

    # Force MJPG codec for higher resolution support
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    print(f"  Camera opened: {actual_w}x{actual_h} @ {actual_fps:.0f} FPS")

    # Start ffmpeg subprocess for H.264 encoding
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{actual_w}x{actual_h}",
        "-r", str(int(actual_fps)),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL)

    # Record loop
    duration_seconds = duration_minutes * 60
    start_time = time.time()
    frame_count = 0
    last_progress = 0
    drop_count = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration_seconds:
                break

            ret, frame = cap.read()
            if not ret:
                drop_count += 1
                if drop_count > 100:
                    print(f"  ERROR: Too many dropped frames ({drop_count}). Stopping.")
                    break
                time.sleep(0.01)
                continue

            ffmpeg_proc.stdin.write(frame.tobytes())
            frame_count += 1
            drop_count = 0

            # Progress every 30 seconds
            progress_interval = int(elapsed // 30)
            if progress_interval > last_progress:
                last_progress = progress_interval
                remaining = duration_seconds - elapsed
                mins_left = int(remaining // 60)
                secs_left = int(remaining % 60)
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0
                print(f"  [{label}] {frame_count} frames | "
                      f"{mins_left}m {secs_left}s remaining | "
                      f"{file_size_mb:.0f} MB written")

    except KeyboardInterrupt:
        print(f"\n  Recording interrupted by user.")

    finally:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        cap.release()

    # Summary
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0
    actual_duration = time.time() - start_time
    print(f"\n  Recording complete:")
    print(f"    Frames:   {frame_count}")
    print(f"    Duration: {actual_duration:.0f}s ({actual_duration / 60:.1f} min)")
    print(f"    File:     {output_path} ({file_size_mb:.0f} MB)")
    print(f"    FPS:      {frame_count / actual_duration:.1f}" if actual_duration > 0 else "")

    return frame_count > 0

def show_schedule():
    """Print the capture schedule."""
    print(f"\n{'=' * 60}")
    print(f"  CAPTURE SCHEDULE")
    print(f"{'=' * 60}")
    for label, hour, minute, duration in CAPTURE_SCHEDULE:
        start = f"{hour:02d}:{minute:02d}"
        end_hour = hour + duration // 60
        end_min = minute + duration % 60
        if end_min >= 60:
            end_hour += 1
            end_min -= 60
        end = f"{end_hour:02d}:{end_min:02d}"
        print(f"  {label:12s}  {start} → {end}  ({duration} min)")
    print(f"\n  Camera:     /dev/video{CAMERA_DEVICE}")
    print(f"  Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS} FPS")
    print(f"  Output dir: {OUTPUT_DIR}/")
    print(f"{'=' * 60}\n")


def run_scheduled(duration_override: int = 0):
    """
    Main scheduling loop. Sleeps between windows, records when it's time.
    Runs indefinitely until Ctrl+C.
    """
    print(f"\nScheduled capture started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Press Ctrl+C to stop.\n")

    show_schedule()

    while True:
        now = datetime.now()
        label, window_start, duration = get_next_window(now)

        if duration_override > 0:
            duration = duration_override

        # Are we inside this window right now?
        window_end = window_start + timedelta(minutes=duration)
        if now >= window_start and now < window_end:
            # We're in a window — record with remaining time
            remaining_minutes = round((window_end - now).total_seconds() / 60)
            if remaining_minutes < 1:
                continue  # skip if less than 1 minutes left in window
            output_path = generate_filename(label, window_start)
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            record_session(label, remaining_minutes, output_path)

            # After recording, find the next window
            continue

        # We're between windows — sleep until the next one
        wait_seconds = (window_start - now).total_seconds()
        if wait_seconds < 0:
            wait_seconds = 0

        wait_hours = int(wait_seconds // 3600)
        wait_mins = int((wait_seconds % 3600) // 60)
        print(f"  Next: {label} at {window_start.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Waiting {wait_hours}h {wait_mins}m...")

        try:
            # Sleep in 60-second intervals so Ctrl+C is responsive
            sleep_end = time.time() + wait_seconds
            while time.time() < sleep_end:
                time.sleep(min(60, sleep_end - time.time()))
        except KeyboardInterrupt:
            print("\nScheduler stopped by user.")
            return


def run_once(duration_override: int = 0):
    """Record one session immediately (for testing)."""
    label = "test"
    duration = duration_override if duration_override > 0 else 5  # default 5 min for test
    output_path = generate_filename(label, datetime.now())
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Single capture mode — recording {duration} minutes immediately.")
    record_session(label, duration, output_path)

    print(f"\nTo run the pipeline on this recording:")
    print(f"  python initialize_pipeline.py --video {output_path} --output data/pipeline_output.jsonl")


def main():
    parser = argparse.ArgumentParser(
        description="Scheduled camera capture for pipeline testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scheduled_capture.py                    # Run full schedule
  python scheduled_capture.py --list             # Show schedule
  python scheduled_capture.py --once             # Quick test recording
  python scheduled_capture.py --once --duration 1  # 1-minute test
  python scheduled_capture.py --duration 30      # Override all windows to 30 min

Edit CAPTURE_SCHEDULE at the top of this file to change times.
Edit CAMERA_DEVICE, FRAME_WIDTH, FRAME_HEIGHT, FPS to change camera settings.
        """,
    )
    parser.add_argument("--once", action="store_true",
                        help="Record one session immediately (skip schedule)")
    parser.add_argument("--duration", type=int, default=0,
                        help="Override recording duration in minutes")
    parser.add_argument("--list", action="store_true",
                        help="Show the capture schedule and exit")
    args = parser.parse_args()

    if args.list:
        show_schedule()
        return

    if args.once:
        run_once(duration_override=args.duration)
    else:
        run_scheduled(duration_override=args.duration)


if __name__ == "__main__":
    main()
