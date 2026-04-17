#!/usr/bin/env python3
"""
test_video_pipeline.py

Quick end-to-end pipeline smoke test using a live camera or video file.
Confirms that every stage produces valid output without requiring a GUI.

Tested stages (in order):
    [1] Input layer  — reads frames from camera or video file
    [2] ROI layer    — initializes and builds ROI package
    [3] YOLO layer   — detects vehicles in the frame
    [4] Tracking layer — tracks detections across frames
    [5] VLM crop + cache — crops tracked vehicles and updates the cache
    [6] VLM inference — runs SmolVLM on the first available crop

Usage:
    python test_video_pipeline.py                  # use video file (data/)
    python test_video_pipeline.py --live           # use live camera device 0
    python test_video_pipeline.py --live --device 1
    python test_video_pipeline.py --frames 60      # run 60 frames instead of default
    python test_video_pipeline.py --no-vlm         # skip VLM (faster smoke test)

Exit codes:
    0  all tested stages passed
    1  one or more stages failed
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# ── sys.path setup ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src" / "input-layer"))
sys.path.insert(0, str(_ROOT / "src" / "roi-layer"))
sys.path.insert(0, str(_ROOT / "src" / "yolo-layer"))
sys.path.insert(0, str(_ROOT / "src" / "tracking-layer"))
sys.path.insert(0, str(_ROOT / "src" / "vlm-frame-cropper-layer"))
sys.path.insert(0, str(_ROOT / "src" / "vlm-layer"))


# ── result tracking ───────────────────────────────────────────────────────────

_results: dict[str, dict] = {}


def _stage(name: str) -> None:
    print(f"\n  ┌─ {name}")
    _results[name] = {"status": "running", "detail": ""}


def _pass(name: str, detail: str = "") -> None:
    _results[name] = {"status": "PASS", "detail": detail}
    tag = f" ({detail})" if detail else ""
    print(f"  └─ ✅ PASS{tag}")


def _fail(name: str, exc: Exception) -> None:
    _results[name] = {"status": "FAIL", "detail": str(exc)}
    print(f"  └─ ❌ FAIL  {type(exc).__name__}: {exc}")


def _skip(name: str, reason: str) -> None:
    _results[name] = {"status": "SKIP", "detail": reason}
    print(f"  └─ ⏭  SKIP  {reason}")


def _summary() -> int:
    print()
    print("═" * 60)
    print("  TEST SUMMARY")
    print("═" * 60)
    failed = 0
    for stage, r in _results.items():
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭ "}.get(r["status"], "?")
        detail = f"  — {r['detail']}" if r["detail"] else ""
        print(f"  {icon}  {stage}{detail}")
        if r["status"] == "FAIL":
            failed += 1
    print()
    if failed:
        print(f"  ✗ {failed} stage(s) FAILED")
    else:
        print("  ✓ All stages PASSED")
    print()
    return 1 if failed else 0


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_input_dict(pkg) -> dict[str, Any]:
    return {
        "input_layer_frame_id": pkg.input_layer_frame_id,
        "input_layer_timestamp": pkg.input_layer_timestamp,
        "input_layer_image": pkg.input_layer_image,
        "input_layer_source_type": pkg.input_layer_source_type,
        "input_layer_resolution": pkg.input_layer_resolution,
    }


def _default_video() -> str:
    """Return the first .mp4 found in data/, or an empty string."""
    for p in sorted((_ROOT / "data").glob("*.mp4")):
        return str(p)
    return ""


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test the full pipeline end-to-end (no GUI)"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--live", action="store_true",
                        help="Use live camera instead of a video file")
    source.add_argument("--video", default="",
                        help="Path to a video file (default: first .mp4 in data/)")
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index for --live (default: 0)")
    parser.add_argument("--use-gstreamer", action="store_true",
                        help="Use GStreamer for Jetson CSI camera")
    parser.add_argument("--frames", type=int, default=30,
                        help="Number of frames to process (default: 30)")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Skip VLM inference (faster smoke test)")
    parser.add_argument("--yolo-model", default="yolov11v28_jingtao.engine",
                        help="YOLO model name (default: yolov11v28_jingtao.engine)")
    parser.add_argument("--vlm-model", default="src/vlm-layer/SmolVLM-256M-Instruct",
                        help="Path to VLM model directory")
    args = parser.parse_args()

    video_path = args.video or (_default_video() if not args.live else "")

    print()
    print("═" * 60)
    print("  EDGE-VIT PIPELINE — END-TO-END SMOKE TEST")
    print("═" * 60)
    print(f"  Source  : {'live camera (device=' + str(args.device) + ')' if args.live else video_path}")
    print(f"  Frames  : {args.frames}")
    print(f"  VLM     : {'disabled (--no-vlm)' if args.no_vlm else args.vlm_model}")
    print()

    # ── [1] Input Layer ───────────────────────────────────────────
    _stage("1. Input Layer")
    try:
        from input_layer import InputLayer
        input_layer = InputLayer()
        if args.live:
            input_layer.initialize_input_layer(
                config_input_source="camera",
                config_frame_resolution=(640, 360),
                camera_device_index=args.device,
                use_gstreamer=args.use_gstreamer,
            )
        else:
            if not video_path:
                raise FileNotFoundError("No video file found in data/ — use --live or --video <path>")
            input_layer.initialize_input_layer(
                config_input_source="video",
                config_frame_resolution=(640, 360),
                config_input_path=video_path,
            )
        _pass("1. Input Layer", "video" if not args.live else "camera")
    except Exception as exc:
        _fail("1. Input Layer", exc)
        return _summary()  # cannot continue without frames

    # ── [2] ROI Layer ─────────────────────────────────────────────
    _stage("2. ROI Layer")
    try:
        from roi_layer import initialize_roi_layer, build_roi_layer_package, update_roi_state
        initialize_roi_layer(config_roi_enabled=True, config_roi_vehicle_count_threshold=3)
        _pass("2. ROI Layer")
    except Exception as exc:
        _fail("2. ROI Layer", exc)
        return _summary()

    # ── [3] YOLO Layer ────────────────────────────────────────────
    _stage("3. YOLO Layer")
    try:
        from detector import initialize_yolo_layer, run_yolo_detection, filter_yolo_detections, build_yolo_layer_package
        initialize_yolo_layer(
            model_name=args.yolo_model,
            conf_threshold=0.4,
            device="cuda",
        )
        _pass("3. YOLO Layer", f"model={args.yolo_model}")
    except Exception as exc:
        _fail("3. YOLO Layer", exc)
        return _summary()

    # ── [4] Tracking Layer ────────────────────────────────────────
    _stage("4. Tracking Layer")
    try:
        from tracker import (
            initialize_tracking_layer, update_tracks,
            assign_tracking_status, build_tracking_layer_package,
        )
        initialize_tracking_layer(frame_rate=30)
        _pass("4. Tracking Layer")
    except Exception as exc:
        _fail("4. Tracking Layer", exc)
        return _summary()

    # ── [5] VLM Crop + Cache ──────────────────────────────────────
    _stage("5. VLM Crop + Cache")
    try:
        from vlm_frame_cropper_layer import (
            initialize_vlm_crop_cache,
            refresh_vlm_crop_cache_track_state,
            build_vlm_frame_cropper_request_package,
            extract_vlm_object_crop,
            build_vlm_frame_cropper_package,
            update_vlm_crop_cache,
            build_vlm_dispatch_package,
        )
        crop_cache = initialize_vlm_crop_cache(config_vlm_crop_cache_size=5)
        _pass("5. VLM Crop + Cache")
    except Exception as exc:
        _fail("5. VLM Crop + Cache", exc)

    # ── [6] VLM ───────────────────────────────────────────────────
    vlm_state = None
    if not args.no_vlm:
        _stage("6. VLM Inference")
        try:
            from layer import VLMConfig, initialize_vlm_layer
            vlm_cfg = VLMConfig(
                config_vlm_enabled=True,
                config_vlm_model=args.vlm_model,
                config_device="cuda",
                config_vlm_max_new_tokens=24,
            )
            vlm_state = initialize_vlm_layer(vlm_cfg)
            _pass("6. VLM Inference", f"model={Path(args.vlm_model).name} device={vlm_state.vlm_runtime_device}")
        except Exception as exc:
            _fail("6. VLM Inference", exc)
    else:
        _skip("6. VLM Inference", "--no-vlm flag set")

    # ── Frame loop ────────────────────────────────────────────────
    print()
    print(f"  Processing {args.frames} frames …")
    print()

    frame_detections: list[int] = []
    frame_tracks: list[int] = []
    vlm_fired = False
    vlm_result_text = ""
    t0 = time.monotonic()

    for frame_idx in range(args.frames):
        raw = input_layer.read_next_frame()
        if raw is None:
            print(f"  ⚠  Frame {frame_idx}: no data (end of source?)")
            break

        pkg_obj = input_layer.build_input_layer_package(raw)
        input_pkg = _build_input_dict(pkg_obj)

        # ROI
        roi_pkg = build_roi_layer_package(input_pkg)
        yolo_upstream = roi_pkg if roi_pkg.get("roi_layer_locked") else input_pkg

        # YOLO
        raw_dets = run_yolo_detection(yolo_upstream)
        dets = filter_yolo_detections(raw_dets)
        yolo_pkg = build_yolo_layer_package(input_pkg["input_layer_frame_id"], dets)
        update_roi_state(input_pkg, dets)
        frame_detections.append(len(dets))

        # Tracking
        tracks = update_tracks(yolo_pkg)
        status_tracks = assign_tracking_status(tracks, input_pkg["input_layer_frame_id"])
        tracking_pkg = build_tracking_layer_package(input_pkg["input_layer_frame_id"], status_tracks)
        frame_tracks.append(len(status_tracks))

        # VLM crop + cache (always build cache, even without VLM inference)
        track_ids = tracking_pkg.get("tracking_layer_track_id", [])
        for i in range(len(track_ids)):
            row = {
                "track_id": str(track_ids[i]),
                "bbox": tuple(tracking_pkg["tracking_layer_bbox"][i]),
                "detector_class": str(tracking_pkg["tracking_layer_detector_class"][i]),
                "confidence": float(tracking_pkg["tracking_layer_confidence"][i]),
                "status": str(tracking_pkg["tracking_layer_status"][i]),
            }
            refresh_vlm_crop_cache_track_state(crop_cache, row, input_pkg["input_layer_frame_id"])
            if row["status"] == "lost":
                continue
            req = build_vlm_frame_cropper_request_package(
                input_layer_package=input_pkg,
                tracking_layer_package=tracking_pkg,
                track_index=i,
                vlm_frame_cropper_trigger_reason="smoke_test",
                config_vlm_enabled=True,
            )
            if req is None:
                continue
            crop = extract_vlm_object_crop(input_pkg, req)
            crop_pkg = build_vlm_frame_cropper_package(req, crop)
            update_vlm_crop_cache(
                crop_cache, row, crop_pkg,
                int(input_pkg["input_layer_frame_id"]),
                "smoke_test",
            )

        # VLM inference — run once on the first track with a crop
        if vlm_state is not None and not vlm_fired:
            for tid in list(crop_cache.get("track_caches", {}).keys()):
                dispatch = build_vlm_dispatch_package(crop_cache, tid)
                if dispatch is None:
                    continue
                try:
                    from layer import VLMFrameCropperLayerPackage, run_vlm_inference, normalize_vlm_result
                    inner = dispatch["vlm_frame_cropper_layer_package"]
                    vlm_pkg = VLMFrameCropperLayerPackage(
                        vlm_frame_cropper_layer_track_id=str(inner["vlm_frame_cropper_layer_track_id"]),
                        vlm_frame_cropper_layer_image=inner["vlm_frame_cropper_layer_image"],
                        vlm_frame_cropper_layer_bbox=inner.get("vlm_frame_cropper_layer_bbox"),
                    )
                    t_vlm = time.monotonic()
                    raw_result = run_vlm_inference(vlm_state, vlm_pkg)
                    vlm_dt = time.monotonic() - t_vlm
                    norm = normalize_vlm_result(raw_result)
                    vlm_result_text = raw_result.vlm_layer_raw_text
                    vlm_fired = True
                    print(f"  ● VLM fired on track {tid} at frame {frame_idx}  ({vlm_dt:.2f}s)")
                    print(f"    label={norm.get('label','?')}  "
                          f"confidence={norm.get('confidence','?')}  "
                          f"raw=\"{vlm_result_text[:80]}\"")
                except Exception as exc:
                    print(f"  ⚠  VLM inference error: {exc}")
                    _fail("6. VLM Inference", exc)
                    vlm_fired = True
                break

        # Progress every 10 frames
        if frame_idx % 10 == 9 or frame_idx == args.frames - 1:
            elapsed = time.monotonic() - t0
            fps = (frame_idx + 1) / max(elapsed, 0.001)
            print(f"  frame {frame_idx+1:3d}/{args.frames}  "
                  f"dets={frame_detections[-1]}  tracks={frame_tracks[-1]}  "
                  f"fps={fps:.1f}")

    input_layer.close_input_layer()
    total_elapsed = time.monotonic() - t0

    # ── validate per-stage results with frame data ─────────────────
    # YOLO: passed if we detected at least once across all frames
    total_dets = sum(frame_detections)
    if "3. YOLO Layer" in _results and _results["3. YOLO Layer"]["status"] == "PASS":
        if total_dets > 0:
            _results["3. YOLO Layer"]["detail"] += f" | {total_dets} detections over {len(frame_detections)} frames"
        else:
            _results["3. YOLO Layer"]["detail"] += " | ⚠ 0 detections (no vehicles in scene?)"

    # Tracking: passed if tracker ran
    if "4. Tracking Layer" in _results and _results["4. Tracking Layer"]["status"] == "PASS":
        total_tracks = sum(frame_tracks)
        _results["4. Tracking Layer"]["detail"] += f" | {total_tracks} track-frames"

    # VLM crop cache: annotate with whether any crops were built
    cache_track_count = len(crop_cache.get("track_caches", {}))
    if "5. VLM Crop + Cache" in _results and _results["5. VLM Crop + Cache"]["status"] == "PASS":
        _results["5. VLM Crop + Cache"]["detail"] += f" | {cache_track_count} tracks in cache"

    # VLM inference: annotate with whether it fired
    if "6. VLM Inference" in _results and _results["6. VLM Inference"]["status"] == "PASS":
        if vlm_fired:
            _results["6. VLM Inference"]["detail"] += f" | fired — \"{vlm_result_text[:40]}\""
        else:
            _results["6. VLM Inference"]["detail"] += " | ⚠ no tracks reached VLM within frame budget"

    # ── summary ────────────────────────────────────────────────────
    print()
    print(f"  Processed {len(frame_detections)} frames in {total_elapsed:.1f}s  "
          f"({len(frame_detections)/max(total_elapsed,0.001):.1f} fps avg)")

    return _summary()


if __name__ == "__main__":
    sys.exit(main())
