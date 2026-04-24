#!/usr/bin/env python3
"""
visualize_vlm_realtime.py
Non-blocking VLM loop visualizer that keeps the video feed moving while VLM
inference runs on a background worker.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from visualize_vlm import (  # noqa: E402
    DISPLAY_SCALE,
    VLMConfig,
    VLMFrameCropperLayerPackage,
    build_canvas,
    build_vlm_ack_package,
    build_vlm_ack_package_from_result,
    build_vlm_frame_cropper_package,
    build_vlm_frame_cropper_request_package,
    build_vlm_layer_package,
    build_tracking_layer_package,
    build_yolo_layer_package,
    draw_text,
    draw_text_block,
    extract_vlm_object_crop,
    filter_yolo_detections,
    get_vehicle_state_record,
    initialize_tracking_layer,
    initialize_vehicle_state_layer,
    initialize_vlm_crop_cache,
    initialize_vlm_layer,
    initialize_yolo_layer,
    input_package_to_dict,
    load_runtime_settings,
    normalize_vlm_result,
    pick_focus_track,
    prepare_vlm_prompt,
    probe_video_metadata,
    refresh_vlm_crop_cache_track_state,
    register_vlm_ack_package,
    run_vlm_inference_batch,
    run_yolo_detection,
    tracking_rows,
    update_tracks,
    update_vehicle_state_from_tracking,
    update_vehicle_state_from_vlm,
    update_vehicle_state_from_vlm_ack,
    update_vlm_crop_cache,
)
from input_layer import InputLayer  # noqa: E402
from tracker import assign_tracking_status  # noqa: E402
from vlm_frame_cropper_layer import build_vlm_dispatch_package  # noqa: E402
from vlm_deferred_queue import (  # noqa: E402
    DeferredVLMTask,
    append_deferred_task,
    encode_crop_image_to_png_base64,
)


class AsyncVLMWorker:
    def __init__(
        self,
        vlm_state: Any,
        feedback_enabled: bool,
        max_queue_size: int = 64,
        batch_size: int = 1,
        batch_wait_ms: int = 20,
        spill_queue_path: str = "",
        spill_max_file_bytes: int = 0,
    ) -> None:
        self.vlm_state = vlm_state
        self.feedback_enabled = feedback_enabled
        self.task_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max_queue_size)
        self.result_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="vlm-worker", daemon=True)
        self.active_task: dict[str, Any] | None = None
        self.completed_count = 0
        self.total_runtime_sec = 0.0
        self.max_runtime_sec = 0.0
        self.batch_size = max(1, int(batch_size))
        self.batch_wait_ms = max(0, int(batch_wait_ms))
        self.spill_queue_path = str(spill_queue_path or "").strip()
        self.spill_max_file_bytes = max(0, int(spill_max_file_bytes))
        self.spilled_count = 0
        self.spill_errors = 0

    def start(self) -> None:
        self.thread.start()

    def shutdown(self, join_timeout: float | None = 30.0) -> None:
        if not self.thread.is_alive():
            return

        inserted_sentinel = False
        while self.thread.is_alive() and not inserted_sentinel:
            try:
                self.task_queue.put(None, timeout=0.1)
                inserted_sentinel = True
            except queue.Full:
                continue

        self.thread.join(timeout=join_timeout)
        if self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=2.0)

    def submit(self, task: dict[str, Any]) -> None:
        try:
            self.task_queue.put_nowait(task)
        except queue.Full:
            if not self.spill_queue_path:
                raise
            self._spill_task(task)

    def drain_results(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        while True:
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def get_status(self) -> dict[str, Any]:
        avg_runtime = self.total_runtime_sec / self.completed_count if self.completed_count else 0.0
        active_track_id = self.active_task["track_id"] if self.active_task else None
        active_frame_id = self.active_task["dispatch_frame_id"] if self.active_task else None
        active_started_at = self.active_task["started_at"] if self.active_task else None
        active_elapsed = time.time() - active_started_at if active_started_at else 0.0
        return {
            "queue_size": self.task_queue.qsize(),
            "busy": self.active_task is not None,
            "active_track_id": active_track_id,
            "active_frame_id": active_frame_id,
            "active_elapsed_sec": active_elapsed,
            "completed_count": self.completed_count,
            "avg_runtime_sec": avg_runtime,
            "max_runtime_sec": self.max_runtime_sec,
            "batch_size": self.batch_size,
            "spilled_count": self.spilled_count,
            "spill_errors": self.spill_errors,
        }

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if task is None:
                break

            batch_tasks = [task]
            batch_start = time.time()
            deadline = batch_start + (self.batch_wait_ms / 1000.0)
            while len(batch_tasks) < self.batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    next_task = self.task_queue.get(timeout=min(0.05, remaining))
                except queue.Empty:
                    continue
                if next_task is None:
                    self.stop_event.set()
                    break
                batch_tasks.append(next_task)

            for t in batch_tasks:
                t["started_at"] = batch_start
            self.active_task = batch_tasks[0]

            try:
                pkgs = [t["vlm_crop_pkg"] for t in batch_tasks]
                qtypes = [t["query_type"] for t in batch_tasks]
                raw_results = run_vlm_inference_batch(self.vlm_state, pkgs, qtypes)

                normalized_results = [normalize_vlm_result(raw) for raw in raw_results]
                if self.feedback_enabled:
                    acks = [build_vlm_ack_package_from_result(raw) for raw in raw_results]
                else:
                    acks = [
                        build_vlm_ack_package(str(t["track_id"]), "accepted", "feedback_disabled_single_shot", False)
                        for t in batch_tasks
                    ]
                error_texts = [""] * len(batch_tasks)
            except Exception as exc:
                raw_results = [None] * len(batch_tasks)
                normalized_results = [None] * len(batch_tasks)
                acks = [
                    build_vlm_ack_package(str(t["track_id"]), "retry_requested", f"VLM error: {exc}", True)
                    for t in batch_tasks
                ]
                error_texts = [str(exc)] * len(batch_tasks)

            runtime_sec = time.time() - batch_start
            for t, raw_result, normalized, ack, error_text in zip(
                batch_tasks, raw_results, normalized_results, acks, error_texts, strict=True
            ):
                self.completed_count += 1
                self.total_runtime_sec += runtime_sec
                self.max_runtime_sec = max(self.max_runtime_sec, runtime_sec)
                self.result_queue.put(
                    {
                        "track_id": str(t["track_id"]),
                        "dispatch_frame_id": int(t["dispatch_frame_id"]),
                        "submitted_at": float(t["submitted_at"]),
                        "prompt": t["prompt_text"],
                        "raw_result": raw_result,
                        "normalized_result": normalized,
                        "ack": ack,
                        "runtime_sec": runtime_sec,
                        "error_text": error_text,
                        "query_type": t["query_type"],
                    }
                )
            self.active_task = None

    def _spill_task(self, task: dict[str, Any]) -> None:
        try:
            crop_pkg = task.get("vlm_crop_pkg")
            crop_img = crop_pkg.vlm_frame_cropper_layer_image if crop_pkg is not None else None
            encoded = encode_crop_image_to_png_base64(crop_img) if crop_img is not None else ""
            append_deferred_task(
                self.spill_queue_path,
                DeferredVLMTask(
                    track_id=str(task.get("track_id", "")),
                    dispatch_frame_id=int(task.get("dispatch_frame_id", -1)),
                    query_type=str(task.get("query_type", "")),
                    prompt_text=str(task.get("prompt_text", "")),
                    crop_png_base64=encoded,
                    bbox=getattr(crop_pkg, "vlm_frame_cropper_layer_bbox", None) if crop_pkg is not None else None,
                    created_at_unix_s=float(task.get("submitted_at", time.time())),
                ),
                max_file_bytes=(self.spill_max_file_bytes if self.spill_max_file_bytes > 0 else None),
            )
            self.spilled_count += 1
            ack = build_vlm_ack_package(str(task.get("track_id", "")), "accepted", "deferred_spill_to_queue", False)
            self.result_queue.put(
                {
                    "track_id": str(task.get("track_id", "")),
                    "dispatch_frame_id": int(task.get("dispatch_frame_id", -1)),
                    "submitted_at": float(task.get("submitted_at", time.time())),
                    "prompt": str(task.get("prompt_text", "")),
                    "raw_result": None,
                    "normalized_result": None,
                    "ack": ack,
                    "runtime_sec": 0.0,
                    "error_text": "deferred_spill_to_queue",
                    "query_type": str(task.get("query_type", "")),
                }
            )
        except Exception as exc:
            self.spill_errors += 1
            raise RuntimeError(f"Failed to spill deferred VLM task: {exc}") from exc


def overlay_async_status(canvas, worker_status: dict[str, Any], current_frame_id: int) -> None:
    x = canvas.shape[1] - 310
    y = 16
    cv2.rectangle(canvas, (x, y), (canvas.shape[1] - 14, y + 134), (25, 25, 32), -1)
    draw_text(canvas, "ASYNC VLM STATUS", x + 10, y + 22, scale=0.42)
    lines = [
        f"queue={worker_status['queue_size']}  busy={worker_status['busy']}",
        f"batch={worker_status.get('batch_size', 1)}  spilled={worker_status.get('spilled_count', 0)}",
        f"active_track={worker_status['active_track_id'] or 'none'}",
        f"active_elapsed={worker_status['active_elapsed_sec']:.1f}s",
        f"completed={worker_status['completed_count']} avg={worker_status['avg_runtime_sec']:.2f}s max={worker_status['max_runtime_sec']:.2f}s",
    ]
    if worker_status["active_frame_id"] is not None:
        lag_frames = max(0, int(current_frame_id) - int(worker_status["active_frame_id"]))
        lines.append(f"active_frame_lag={lag_frames}")
    draw_text_block(canvas, lines, (x + 10, y + 48), scale=0.31, line_gap=16)


def maybe_sleep_to_source_fps(start_wall: float, frame_id: int, fps: float) -> None:
    if fps <= 0:
        return
    target_elapsed = frame_id / fps
    sleep_sec = target_elapsed - (time.time() - start_wall)
    if sleep_sec > 0:
        time.sleep(sleep_sec)


def main() -> None:
    defaults = load_runtime_settings()
    parser = argparse.ArgumentParser(description="Non-blocking VLM visualizer with background inference worker")
    parser.add_argument("--input-source", default=defaults["input_source"], choices=["video", "camera"])
    parser.add_argument("--video", default=defaults["video"])
    parser.add_argument("--model", default=defaults["model"])
    parser.add_argument("--conf", type=float, default=defaults["conf"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--vlm-device", default="", help="Override device for VLM only")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--gstreamer", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", default="", help="Optional output video path")
    parser.add_argument(
        "--vlm-mode",
        default=str(defaults.get("vlm_runtime_mode", "inline")).strip().lower(),
        choices=["inline", "async", "spill"],
        help="VLM dispatch mode: inline|async|spill (spill writes overflow to queue file).",
    )
    parser.add_argument("--max-queue-size", type=int, default=int(defaults.get("vlm_worker_max_queue_size", 64)))
    parser.add_argument(
        "--vlm-batch-size",
        type=int,
        default=int(defaults.get("vlm_worker_batch_size", 1)),
        help="Micro-batch size for VLM worker (1 disables batching).",
    )
    parser.add_argument(
        "--vlm-batch-wait-ms",
        type=int,
        default=int(defaults.get("vlm_worker_batch_wait_ms", 20)),
        help="Max wait to form a batch before running VLM.",
    )
    parser.add_argument(
        "--vlm-spill-queue",
        default=str(defaults.get("vlm_worker_spill_queue_path", "") or ""),
        help="JSONL path used by spill mode (cache-and-run).",
    )
    parser.add_argument(
        "--spill-max-file-mb",
        type=float,
        default=float(defaults.get("vlm_spill_max_file_mb", 0.0) or 0.0),
        help="Rotate spill JSONL when file size reaches this many MB (0 = unlimited).",
    )
    parser.add_argument(
        "--no-realtime-throttle",
        action="store_true",
        default=not bool(defaults.get("vlm_realtime_throttle_enabled", True)),
        help="Run as fast as possible instead of pacing to source FPS",
    )
    args = parser.parse_args()

    frame_resolution = tuple(defaults["frame_resolution"])
    fps, width, height = probe_video_metadata(args.input_source, args.video, frame_resolution)
    output_w = int(width * DISPLAY_SCALE) + 780
    output_h = int(height * DISPLAY_SCALE)
    out = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps or 30.0, (output_w, output_h))
        if not out.isOpened():
            raise RuntimeError(f"Could not create output video: {args.output}")

    vlm_device = args.vlm_device.strip() or args.device
    vlm_state = initialize_vlm_layer(
        VLMConfig(
            config_vlm_enabled=bool(defaults["vlm_enabled"]),
            config_vlm_model=defaults["vlm_model"],
            config_device=vlm_device,
        )
    )

    input_layer = InputLayer()
    input_layer.initialize_input_layer(
        config_input_source=args.input_source,
        config_frame_resolution=frame_resolution,
        config_input_path=args.video,
        camera_device_index=args.camera_index,
        use_gstreamer=args.gstreamer,
    )
    initialize_yolo_layer(model_name=args.model, conf_threshold=args.conf, device=args.device)
    initialize_tracking_layer(frame_rate=int(fps))
    crop_cache_state = initialize_vlm_crop_cache(defaults["vlm_crop_cache_size"], defaults["vlm_dead_after_lost_frames"])
    initialize_vehicle_state_layer(prune_after_lost_frames=None)
    spill_queue_path = args.vlm_spill_queue if args.vlm_mode == "spill" else ""
    spill_mb = float(args.spill_max_file_mb)
    spill_max_bytes = int(spill_mb * 1024 * 1024) if spill_mb > 0.0 else 0
    worker = AsyncVLMWorker(
        vlm_state=vlm_state,
        feedback_enabled=bool(defaults["vlm_crop_feedback_enabled"]),
        max_queue_size=args.max_queue_size,
        batch_size=args.vlm_batch_size,
        batch_wait_ms=args.vlm_batch_wait_ms,
        spill_queue_path=spill_queue_path,
        spill_max_file_bytes=spill_max_bytes,
    )
    worker.start()

    debug_state: dict[str, dict[str, Any]] = {}
    start_wall = time.time()

    try:
        while True:
            for result in worker.drain_results():
                ack = result["ack"]
                register_vlm_ack_package(crop_cache_state, ack)
                update_vehicle_state_from_vlm_ack(ack)
                if ack.vlm_ack_status == "accepted" and result["raw_result"] is not None:
                    update_vehicle_state_from_vlm(build_vlm_layer_package(result["raw_result"]))
                debug_state[result["track_id"]] = {
                    "prompt": result["prompt"],
                    "raw_response": result["raw_result"].vlm_layer_raw_text.strip() if result["raw_result"] else f"ERROR: {result['error_text']}",
                    "normalized_result": result["normalized_result"],
                    "ack_status": ack.vlm_ack_status,
                    "ack_reason": ack.vlm_ack_reason,
                    "retry_reasons": (result["normalized_result"] or {}).get("vlm_retry_reasons", []) if defaults["vlm_crop_feedback_enabled"] else [],
                    "progressed_only_tracking": ack.vlm_ack_status == "accepted",
                    "runtime_sec": result["runtime_sec"],
                    "dispatch_frame_id": result["dispatch_frame_id"],
                    "queue_delay_sec": max(0.0, result["runtime_sec"] + (result["submitted_at"] - start_wall)),
                    "query_type": result["query_type"],
                }

            raw_frame = input_layer.read_next_frame()
            if raw_frame is None:
                break

            input_package = input_layer.build_input_layer_package(raw_frame)
            input_pkg = input_package_to_dict(input_package)
            raw_dets = run_yolo_detection(input_pkg)
            filtered_dets = filter_yolo_detections(raw_dets)
            yolo_pkg = build_yolo_layer_package(input_package.input_layer_frame_id, filtered_dets)
            current_tracks = update_tracks(yolo_pkg)
            status_tracks = assign_tracking_status(current_tracks, input_package.input_layer_frame_id)
            tracking_pkg = build_tracking_layer_package(input_package.input_layer_frame_id, status_tracks)
            update_vehicle_state_from_tracking(tracking_pkg)

            frame_view = input_package.input_layer_image.copy()
            dispatch_track_ids: list[str] = []

            for index, row in enumerate(tracking_rows(tracking_pkg)):
                refresh_vlm_crop_cache_track_state(crop_cache_state, row, input_package.input_layer_frame_id)
                vehicle_record_for_row = get_vehicle_state_record(row["track_id"])
                if vehicle_record_for_row and vehicle_record_for_row.get("vehicle_state_layer_vlm_called"):
                    debug_state.setdefault(str(row["track_id"]), {})["progressed_only_tracking"] = True
                    continue
                if row["status"] == "lost":
                    continue
                request_pkg = build_vlm_frame_cropper_request_package(
                    input_layer_package=input_package,
                    tracking_layer_package=tracking_pkg,
                    track_index=index,
                    vlm_frame_cropper_trigger_reason=f'tracking_status:{row["status"]}',
                    config_vlm_enabled=True,
                )
                if request_pkg is None:
                    continue
                crop = extract_vlm_object_crop(input_package, request_pkg)
                crop_pkg = build_vlm_frame_cropper_package(request_pkg, crop)
                update_vlm_crop_cache(crop_cache_state, row, crop_pkg, input_package.input_layer_frame_id, f'tracking_status:{row["status"]}')

            if vlm_state.config_vlm_enabled:
                for tid in sorted(crop_cache_state["track_caches"].keys(), key=lambda x: (len(str(x)), str(x))):
                    vehicle_record_for_dispatch = get_vehicle_state_record(tid)
                    if vehicle_record_for_dispatch and vehicle_record_for_dispatch.get("vehicle_state_layer_vlm_called"):
                        continue
                    dispatch = build_vlm_dispatch_package(crop_cache_state, tid)
                    if dispatch is None:
                        continue
                    dispatch_track_ids.append(str(tid))
                    inner = dispatch["vlm_frame_cropper_layer_package"]
                    vlm_crop_pkg = VLMFrameCropperLayerPackage(
                        vlm_frame_cropper_layer_track_id=str(inner["vlm_frame_cropper_layer_track_id"]),
                        vlm_frame_cropper_layer_image=inner["vlm_frame_cropper_layer_image"],
                        vlm_frame_cropper_layer_bbox=tuple(int(x) for x in inner["vlm_frame_cropper_layer_bbox"]),
                    )
                    query_type = DEFAULT_VLM_QUERY_TYPE if defaults["vlm_crop_feedback_enabled"] else "vehicle_semantics_single_shot_v1"
                    if dispatch["vlm_dispatch_mode"] == "dead_best_available":
                        query_type = "vehicle_semantics_single_shot_v1"
                    worker.submit(
                        {
                            "track_id": str(tid),
                            "dispatch_frame_id": int(input_package.input_layer_frame_id),
                            "prompt_text": prepare_vlm_prompt(query_type, vlm_crop_pkg),
                            "query_type": query_type,
                            "submitted_at": time.time(),
                            "vlm_crop_pkg": vlm_crop_pkg,
                        }
                    )

            focus_track_id = pick_focus_track(crop_cache_state, dispatch_track_ids)
            vehicle_record = get_vehicle_state_record(focus_track_id) if focus_track_id is not None else None
            elapsed = time.time() - start_wall
            fps_text = input_package.input_layer_frame_id / elapsed if elapsed > 0 else 0.0
            canvas = build_canvas(
                frame_view,
                tracking_pkg,
                focus_track_id,
                crop_cache_state,
                debug_state,
                vehicle_record,
                defaults["vlm_crop_cache_size"],
                fps_text,
                input_package.input_layer_frame_id,
                bool(defaults["vlm_crop_feedback_enabled"]),
            )
            overlay_async_status(canvas, worker.get_status(), input_package.input_layer_frame_id)

            if out is not None:
                out.write(canvas)
            if args.show:
                cv2.imshow("VLM realtime visualizer", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.max_frames > 0 and input_package.input_layer_frame_id >= args.max_frames:
                break
            if not args.no_realtime_throttle:
                maybe_sleep_to_source_fps(start_wall, input_package.input_layer_frame_id, fps)
    finally:
        worker.shutdown()
        input_layer.close_input_layer()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

    if args.output:
        print(f"Saved VLM realtime visualization to: {args.output}")


if __name__ == "__main__":
    main()
