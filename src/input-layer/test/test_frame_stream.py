#!/usr/bin/env python3
"""
test_frame_stream.py

MS1 deliverable: Frame stream test script (prints frame index).

This script validates that the input layer can:
    1. Initialize from config values
    2. Read frames sequentially
    3. Build properly structured input_layer_package objects
    4. Clean up resources on exit

Usage
-----
    python src/input-layer/test/test_frame_stream.py

Edit the CONFIG section below to change source, path, and resolution.
"""

import os
import sys
import time

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these values before running
# ══════════════════════════════════════════════════════════════════════════════

INPUT_SOURCE  = "camera"        # "camera" or "video"
CAMERA_DEVICE = 0               # device index — /dev/video0 = 0, /dev/video1 = 1, …
RESOLUTION    = (640, 480)      # (width, height) in pixels
VIDEO_PATH    = "data/sample1.mp4"  # only used when INPUT_SOURCE = "video"

SHOW_PREVIEW  = True            # show a live OpenCV window while streaming frames

# ══════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_INPUT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_CONFIG_DIR = os.path.abspath(os.path.join(_INPUT_DIR, "..", "configuration-layer"))
if _INPUT_DIR not in sys.path:
    sys.path.insert(0, _INPUT_DIR)
if _CONFIG_DIR not in sys.path:
    sys.path.insert(0, _CONFIG_DIR)

from input_layer import InputLayer
from input_layer_package import InputLayerPackage
from config_node import load_config, validate_config


# ------------------------------------------------------------------
# ANSI colors for terminal output
# ------------------------------------------------------------------
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def _pass(msg: str) -> None:
    print(f"  {_GREEN}PASS{_RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {_RED}FAIL{_RESET}  {msg}")


def _info(msg: str) -> None:
    print(f"  {_YELLOW}INFO{_RESET}  {msg}")


def run_tests(source: str, path: str, resolution: tuple) -> None:
    passed = 0
    failed = 0

    print("=" * 60)
    print("  INPUT LAYER — FRAME STREAM TEST")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: Config loads and validates
    # ------------------------------------------------------------------
    print("\n[Test 1] Config load & validate")
    try:
        config = load_config({
            "config_input_source": source,
            "config_input_path": path,
            "config_frame_resolution": resolution,
        })
        validate_config(config)
        _pass("Config loaded and validated.")
        passed += 1
    except Exception as exc:
        _fail(f"Config error: {exc}")
        failed += 1
        print(f"\nCannot continue without valid config. Exiting.")
        return

    # ------------------------------------------------------------------
    # Test 2: InputLayer initializes
    # ------------------------------------------------------------------
    print("\n[Test 2] InputLayer initialization")
    layer = InputLayer()
    try:
        layer.initialize_input_layer(
            config_input_source=source,
            config_frame_resolution=resolution,
            config_input_path=path,
            camera_device_index=CAMERA_DEVICE,
        )
        assert layer.is_initialized, "Layer reports not initialized"
        _pass(f"Initialized with source='{source}'.")
        passed += 1
    except Exception as exc:
        _fail(f"Init error: {exc}")
        failed += 1
        print(f"\nCannot continue without initialized layer. Exiting.")
        return

    # ------------------------------------------------------------------
    # Test 3: Read first frame
    # ------------------------------------------------------------------
    print("\n[Test 3] Read first frame")
    raw_frame = layer.read_next_frame()
    if raw_frame is not None:
        _pass(f"Got frame with shape {raw_frame.shape}")
        passed += 1
    else:
        _fail("read_next_frame returned None on first call.")
        failed += 1
        layer.close_input_layer()
        return

    # ------------------------------------------------------------------
    # Test 4: Build package from frame
    # ------------------------------------------------------------------
    print("\n[Test 4] Build input_layer_package")
    pkg = layer.build_input_layer_package(raw_frame)
    checks = [
        ("frame_id == 1", pkg.input_layer_frame_id == 1),
        ("source_type matches", pkg.input_layer_source_type == source),
        ("resolution matches", pkg.input_layer_resolution == resolution),
        ("image shape matches resolution",
         pkg.input_layer_image.shape[1] == resolution[0]
         and pkg.input_layer_image.shape[0] == resolution[1]),
        ("timestamp is recent", abs(time.time() - pkg.input_layer_timestamp) < 5),
    ]
    for label, ok in checks:
        if ok:
            _pass(label)
            passed += 1
        else:
            _fail(label)
            failed += 1

    # ------------------------------------------------------------------
    # Test 5: Stream N frames and print indices
    # ------------------------------------------------------------------
    import cv2
    import os

    n_stream = 10
    preview_label = "test_frame_stream — press Q to stop"

    # Preview requires a local display — not available over SSH
    preview_active = SHOW_PREVIEW and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    print(f"\n[Test 5] Stream {n_stream} more frames (print frame index)")
    if SHOW_PREVIEW and not preview_active:
        _info("No display detected (SSH session?) — preview disabled.")
    elif preview_active:
        _info("Preview window open — press Q to quit early.")

    stream_ok = True
    for i in range(n_stream):
        raw = layer.read_next_frame()
        if raw is None:
            _info(f"Source exhausted at stream frame {i}. This is OK for short videos.")
            break
        pkg = layer.build_input_layer_package(raw)
        print(f"    frame_id={pkg.input_layer_frame_id}  "
              f"resolution={pkg.input_layer_resolution}  "
              f"timestamp={pkg.input_layer_timestamp:.3f}")
        if preview_active:
            cv2.imshow(preview_label, pkg.input_layer_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                _info("Preview closed by user.")
                break
        if pkg.input_layer_frame_id != i + 2:  # +2 because we already read one
            _fail(f"Expected frame_id {i + 2}, got {pkg.input_layer_frame_id}")
            stream_ok = False
            failed += 1
            break
    if preview_active:
        cv2.destroyAllWindows()
    if stream_ok:
        _pass("Frame indices are monotonically increasing.")
        passed += 1

    # ------------------------------------------------------------------
    # Test 6: Clean shutdown
    # ------------------------------------------------------------------
    print("\n[Test 6] Close input layer")
    try:
        layer.close_input_layer()
        assert not layer.is_initialized, "Layer still reports initialized after close"
        _pass("Layer closed cleanly.")
        passed += 1
    except Exception as exc:
        _fail(f"Close error: {exc}")
        failed += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    total = passed + failed
    color = _GREEN if failed == 0 else _RED
    print(f"  {color}{passed}/{total} checks passed{_RESET}")
    print("=" * 60)


def main():
    path = VIDEO_PATH if INPUT_SOURCE == "video" else ""
    run_tests(source=INPUT_SOURCE, path=path, resolution=RESOLUTION)


if __name__ == "__main__":
    main()
