#!/usr/bin/env python3
"""
End-to-end real-time benchmark: YOLO + SmolVLM with optional parallelism.

Tests the SmolVLM-256M implementation in the full pipeline context:
- Measures YOLO detection latency
- Measures VLM query latency (with real YOLO detections as crops)
- Optionally runs YOLO and VLM in parallel threads
- Reports throughput, latency breakdown, GPU memory usage

Usage:
  # Sequential (YOLO → VLM):
  python3 benchmark_smolvlm_realtime.py

  # Parallel (YOLO || VLM):
  PARALLEL=1 python3 benchmark_smolvlm_realtime.py

  # With INT8 quantized model:
  QUANT=1 python3 benchmark_smolvlm_realtime.py

  # Custom config:
  CONFIG=config.jetson.vlm-smolvlm-256m-trt.yaml python3 benchmark_smolvlm_realtime.py
"""

import os
import sys
import time
import threading
from pathlib import Path
from collections import defaultdict
import numpy as np

# Jetson memory setup (before any CUDA init)
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"

CONFIG_DIR = SRC_DIR / "configuration-layer"
INPUT_DIR = SRC_DIR / "input-layer"
YOLO_DIR = SRC_DIR / "yolo-layer"
VLM_DIR = SRC_DIR / "vlm-layer"
VLM_UTIL_DIR = VLM_DIR / "util"
TRACKING_DIR = SRC_DIR / "tracking-layer"
VSTATE_DIR = SRC_DIR / "vehicle-state-layer"

for p in (CONFIG_DIR, INPUT_DIR, YOLO_DIR, TRACKING_DIR, VSTATE_DIR, VLM_DIR, VLM_UTIL_DIR):
    if p.exists():
        sys.path.insert(0, str(p))

# Configuration
CONFIG_OVERRIDE = os.environ.get("CONFIG", "").strip()
USE_PARALLEL = os.environ.get("PARALLEL", "0").strip() == "1"
USE_QUANT = os.environ.get("QUANT", "0").strip() == "1"
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "30"))
QUERY_TEXT = "What vehicle is this? License plate if visible."

print("=" * 70)
print("SMOLVLM-256M REALTIME BENCHMARK")
print("=" * 70)
print(f"  Parallel mode: {'YES' if USE_PARALLEL else 'NO'}")
print(f"  INT8 quantized: {'YES' if USE_QUANT else 'NO'}")
print(f"  Frames to process: {NUM_FRAMES}")
print()


def load_pipeline():
    """Load all pipeline components."""
    from config_node import get_config_value, load_config, validate_config
    from input_layer import InputLayer
    from detector import initialize_yolo_layer
    from layer import initialize_vlm_layer, query_vlm_layer, VLMConfig

    config_dir = REPO_ROOT / "src" / "configuration-layer"
    if CONFIG_OVERRIDE:
        config_path = config_dir / CONFIG_OVERRIDE
    else:
        config_path = config_dir / "config.yaml"

    print(f"[Config] Loading: {config_path.name}")
    cfg = load_config(config_path)
    validate_config(cfg)

    vlm_enabled = bool(get_config_value(cfg, "config_vlm_enabled"))
    vlm_model = str(get_config_value(cfg, "config_vlm_model"))
    
    # Override to INT8 if requested
    if USE_QUANT:
        vlm_model = str(REPO_ROOT / "src" / "vlm-layer" / "SmolVLM-256M-Instruct-int8")
        print(f"[VLM] Using INT8 model: {vlm_model}")

    device = str(get_config_value(cfg, "config_device"))
    yolo_model = str(get_config_value(cfg, "config_yolo_model"))
    conf_thresh = float(get_config_value(cfg, "config_yolo_confidence_threshold"))

    print(f"[YOLO] Model: {yolo_model}, device: {device}, conf: {conf_thresh}")

    yolo_state = initialize_yolo_layer(
        model_path=yolo_model,
        device=device,
        force_pt=False,
        force_onnx=False,
    )
    print(f"[YOLO] ✓ Initialized")

    vlm_state = None
    if vlm_enabled:
        vlm_state = initialize_vlm_layer(
            VLMConfig(
                config_vlm_enabled=True,
                config_vlm_model=vlm_model,
                config_device=device,
            )
        )
        print(f"[VLM] ✓ Initialized: {vlm_state.vlm_runtime_model_id}")

    return cfg, yolo_state, vlm_state


def benchmark_sequential(cfg, yolo_state, vlm_state, frames_list):
    """Run YOLO → VLM sequentially."""
    from detector import run_yolo_detection
    from vlm_layer import query_vlm_layer
    from config_node import get_config_value
    import cv2

    print("\n[Benchmark] SEQUENTIAL mode")
    print("-" * 70)

    conf_thresh = float(get_config_value(cfg, "config_yolo_confidence_threshold"))
    iou_thresh = float(get_config_value(cfg, "config_yolo_iou_threshold"))

    yolo_times = []
    vlm_times = []
    total_times = []

    for frame_idx, frame in enumerate(frames_list, 1):
        t_start = time.time()

        # YOLO detection
        t_yolo_start = time.time()
        detections = run_yolo_detection(
            frame=frame,
            yolo_layer_state=yolo_state,
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh,
        )
        yolo_elapsed = time.time() - t_yolo_start
        yolo_times.append(yolo_elapsed * 1000)

        # VLM query on first detection (if any)
        vlm_elapsed = 0
        if vlm_state and detections and len(detections) > 0:
            # Extract first detection's crop
            det = detections[0]
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                t_vlm_start = time.time()
                result = query_vlm_layer(vlm_state, crop, QUERY_TEXT)
                vlm_elapsed = time.time() - t_vlm_start
                vlm_times.append(vlm_elapsed * 1000)

        total_elapsed = time.time() - t_start
        total_times.append(total_elapsed * 1000)

        if frame_idx % 5 == 0:
            print(
                f"  Frame {frame_idx}: YOLO {yolo_elapsed*1000:.0f}ms, "
                f"VLM {vlm_elapsed*1000:.0f}ms, Total {total_elapsed*1000:.0f}ms"
            )

    return yolo_times, vlm_times, total_times


def benchmark_parallel(cfg, yolo_state, vlm_state, frames_list):
    """Run YOLO and VLM in parallel threads (VLM processes crops from YOLO queue)."""
    from detector import run_yolo_detection
    from vlm_layer import query_vlm_layer
    from config_node import get_config_value
    import cv2
    import queue

    print("\n[Benchmark] PARALLEL mode (YOLO || VLM)")
    print("-" * 70)

    conf_thresh = float(get_config_value(cfg, "config_yolo_confidence_threshold"))
    iou_thresh = float(get_config_value(cfg, "config_yolo_iou_threshold"))

    # Queues for passing data between threads
    crop_queue = queue.Queue(maxsize=2)  # Buffer of 2 crops
    yolo_times = []
    vlm_times = []
    total_times = []
    frame_times = defaultdict(dict)
    lock = threading.Lock()

    def vlm_worker():
        """Background thread: process crops from queue."""
        while True:
            item = crop_queue.get()
            if item is None:  # Sentinel: stop
                break

            frame_idx, crop = item
            if vlm_state and crop.size > 0:
                t_vlm_start = time.time()
                result = query_vlm_layer(vlm_state, crop, QUERY_TEXT)
                vlm_elapsed = time.time() - t_vlm_start

                with lock:
                    vlm_times.append(vlm_elapsed * 1000)
                    frame_times[frame_idx]["vlm"] = vlm_elapsed * 1000

    # Start VLM worker thread
    vlm_thread = threading.Thread(target=vlm_worker, daemon=True)
    vlm_thread.start()

    # Main thread: YOLO only
    for frame_idx, frame in enumerate(frames_list, 1):
        t_start = time.time()

        t_yolo_start = time.time()
        detections = run_yolo_detection(
            frame=frame,
            yolo_layer_state=yolo_state,
            conf_threshold=conf_thresh,
            iou_threshold=iou_thresh,
        )
        yolo_elapsed = time.time() - t_yolo_start

        with lock:
            yolo_times.append(yolo_elapsed * 1000)
            frame_times[frame_idx]["yolo"] = yolo_elapsed * 1000

        # Queue crop for VLM worker
        if detections and len(detections) > 0:
            det = detections[0]
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crop_queue.put((frame_idx, crop))

        total_elapsed = time.time() - t_start
        total_times.append(total_elapsed * 1000)

        if frame_idx % 5 == 0:
            vlm_ms = frame_times[frame_idx].get("vlm", 0)
            print(
                f"  Frame {frame_idx}: YOLO {yolo_elapsed*1000:.0f}ms, "
                f"(VLM {vlm_ms:.0f}ms buffered), Total {total_elapsed*1000:.0f}ms"
            )

    # Wait for VLM worker to finish
    crop_queue.put(None)
    vlm_thread.join(timeout=10)

    return yolo_times, vlm_times, total_times


def print_stats(yolo_times, vlm_times, total_times):
    """Print statistics."""
    import torch

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if yolo_times:
        print(f"\nYOLO Detection:")
        print(f"  Mean: {np.mean(yolo_times):.0f}ms")
        print(f"  Min: {np.min(yolo_times):.0f}ms, Max: {np.max(yolo_times):.0f}ms")
        print(f"  P95: {np.percentile(yolo_times, 95):.0f}ms")

    if vlm_times:
        print(f"\nVLM Query (SmolVLM-256M):")
        print(f"  Mean: {np.mean(vlm_times):.0f}ms")
        print(f"  Min: {np.min(vlm_times):.0f}ms, Max: {np.max(vlm_times):.0f}ms")
        print(f"  P95: {np.percentile(vlm_times, 95):.0f}ms")

    if total_times:
        fps = 1000 / np.mean(total_times)
        print(f"\nTotal Frame Latency:")
        print(f"  Mean: {np.mean(total_times):.0f}ms ({fps:.1f} FPS)")
        print(f"  Min: {np.min(total_times):.0f}ms, Max: {np.max(total_times):.0f}ms")
        print(f"  P95: {np.percentile(total_times, 95):.0f}ms")

    # GPU memory
    try:
        alloc_mb = torch.cuda.memory_allocated(0) / 1024 / 1024
        total_mb = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        print(f"\nGPU Memory:")
        print(f"  Allocated: {alloc_mb:.0f}MB / {total_mb:.0f}MB ({100*alloc_mb/total_mb:.1f}%)")
    except Exception:
        pass

    print("=" * 70)


def load_test_video(num_frames):
    """Load frames from configured video."""
    from config_node import get_config_value, load_config
    import cv2

    config_path = (
        REPO_ROOT / "src" / "configuration-layer" / (CONFIG_OVERRIDE or "config.yaml")
    )
    cfg = load_config(config_path)
    video_path = get_config_value(cfg, "config_input_path")

    if not os.path.exists(video_path):
        # Try repo-relative
        video_path = REPO_ROOT / video_path

    print(f"[Video] Loading from: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"[Video] Loaded {len(frames)} frames")
    return frames


def main():
    import torch

    torch.backends.cudnn.benchmark = True

    try:
        cfg, yolo_state, vlm_state = load_pipeline()
        frames = load_test_video(NUM_FRAMES)

        if USE_PARALLEL:
            yolo_times, vlm_times, total_times = benchmark_parallel(
                cfg, yolo_state, vlm_state, frames
            )
        else:
            yolo_times, vlm_times, total_times = benchmark_sequential(
                cfg, yolo_state, vlm_state, frames
            )

        print_stats(yolo_times, vlm_times, total_times)

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
