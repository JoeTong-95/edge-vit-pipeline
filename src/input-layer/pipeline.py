#!/usr/bin/env python3
"""
pipeline.py

CLI entry point for running the input layer.

Usage
-----
    # From the project root (inside Docker or on host):
    python -m src.input-layer.pipeline --input video --path data/test.mp4
    python -m src.input-layer.pipeline --input camera

    # Or run directly:
    python src/input-layer/pipeline.py --input video --path data/test.mp4
    python src/input-layer/pipeline.py --input camera --gstreamer

MS1 deliverable:
    CLI support: pipeline.py --input <video | camera>
"""

import argparse
import sys
import os

# ---------------------------------------------------------------------------
# Allow running as a standalone script from the project root
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "configuration-layer"))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
if _CONFIG_DIR not in sys.path:
    sys.path.insert(0, _CONFIG_DIR)

from input_layer import InputLayer
from input_layer_package import InputLayerPackage
from config_node import load_config, validate_config, get_config_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edge VLM Pipeline — Input Layer CLI"
    )
    parser.add_argument(
        "--input",
        required=True,
        choices=["video", "camera"],
        help="Frame source type: 'video' for file, 'camera' for live device.",
    )
    parser.add_argument(
        "--path",
        default="",
        help="Path to video file (required when --input is 'video').",
    )
    parser.add_argument(
        "--resolution",
        default="640x480",
        help="Target frame resolution as WIDTHxHEIGHT (default: 640x480).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    parser.add_argument(
        "--gstreamer",
        action="store_true",
        help="Use GStreamer pipeline for Jetson CSI camera.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = run until source ends / Ctrl-C).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse resolution string
    try:
        w, h = (int(x) for x in args.resolution.split("x"))
    except ValueError:
        print(f"ERROR: Invalid resolution format '{args.resolution}'. Use WIDTHxHEIGHT.")
        sys.exit(1)

    # Build config via the configuration layer
    config = load_config({
        "config_input_source": args.input,
        "config_input_path": args.path,
        "config_frame_resolution": (w, h),
    })

    # Validate (will raise if video path is missing / invalid)
    try:
        validate_config(config)
    except (ValueError, FileNotFoundError) as exc:
        print(f"CONFIG ERROR: {exc}")
        sys.exit(1)

    # Initialize input layer
    layer = InputLayer()
    layer.initialize_input_layer(
        config_input_source=get_config_value(config, "config_input_source"),
        config_frame_resolution=get_config_value(config, "config_frame_resolution"),
        config_input_path=get_config_value(config, "config_input_path"),
        camera_device_index=args.camera_index,
        use_gstreamer=args.gstreamer,
    )

    print(f"[input_layer] initialized — source={args.input}, resolution={w}x{h}")
    if args.input == "video":
        print(f"[input_layer] video path: {args.path}")

    # Main read loop
    try:
        while True:
            raw_frame = layer.read_next_frame()
            if raw_frame is None:
                print("[input_layer] source exhausted or read failed — stopping.")
                break

            package = layer.build_input_layer_package(raw_frame)
            print(package)

            if args.max_frames > 0 and package.input_layer_frame_id >= args.max_frames:
                print(f"[input_layer] reached --max-frames={args.max_frames} — stopping.")
                break

    except KeyboardInterrupt:
        print("\n[input_layer] interrupted by user.")

    finally:
        layer.close_input_layer()
        print("[input_layer] closed.")


if __name__ == "__main__":
    main()
