#!/usr/bin/env python3
"""
run_deferred_vlm_queue.py

Offline "cache-and-run" processor for VLM tasks spilled to a JSONL queue.

Input format: lines produced by `vlm_deferred_queue.append_deferred_task(...)`.
Long runs may split spill output across multiple files: the active queue is rotated
to `*.jsonl.rotated.<ms>` when `config_vlm_spill_max_file_mb` is set; process each
file (or glob) through this script as needed.
Output: a JSONL file with normalized + ack packages per task (plus raw text).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_VLM_DIR = _THIS_DIR.parent
for path in (_THIS_DIR, _VLM_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from layer import (
    VLMConfig,
    VLMFrameCropperLayerPackage,
    build_vlm_ack_package_from_result,
    build_vlm_layer_package,
    initialize_vlm_layer,
    normalize_vlm_result,
    run_vlm_inference_batch,
)
from vlm_deferred_queue import decode_crop_image, load_deferred_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process deferred VLM JSONL queue (offline).")
    parser.add_argument("--queue", required=True, help="Path to deferred queue JSONL file.")
    parser.add_argument("--out", required=True, help="Path to output JSONL file.")
    parser.add_argument("--model", required=True, help="Path to local VLM checkpoint directory.")
    parser.add_argument("--device", default="auto", help="Device for VLM (auto|cpu|cuda).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N tasks (0 = all).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue_path = Path(args.queue)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_deferred_tasks(queue_path, limit=(args.limit or None))
    if not tasks:
        print(f"No tasks found at: {queue_path}")
        return

    vlm_state = initialize_vlm_layer(
        VLMConfig(
            config_vlm_enabled=True,
            config_vlm_model=str(Path(args.model).expanduser()),
            config_device=str(args.device),
        )
    )

    batch_size = max(1, int(args.batch_size))
    processed = 0
    started = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        for base in range(0, len(tasks), batch_size):
            batch = tasks[base: base + batch_size]
            pkgs = [
                VLMFrameCropperLayerPackage(
                    vlm_frame_cropper_layer_track_id=task.track_id,
                    vlm_frame_cropper_layer_image=decode_crop_image(task.crop_png_base64),
                    vlm_frame_cropper_layer_bbox=task.bbox,
                )
                for task in batch
            ]
            query_types = [task.query_type for task in batch]
            raw_results = run_vlm_inference_batch(vlm_state, pkgs, query_types)
            for task, raw in zip(batch, raw_results, strict=True):
                normalized = normalize_vlm_result(raw)
                ack = build_vlm_ack_package_from_result(raw)
                layer_pkg = build_vlm_layer_package(raw)
                f.write(
                    json.dumps(
                        {
                            "track_id": task.track_id,
                            "dispatch_frame_id": task.dispatch_frame_id,
                            "query_type": task.query_type,
                            "model_id": raw.vlm_layer_model_id,
                            "created_at_unix_s": task.created_at_unix_s,
                            "processed_at_unix_s": time.time(),
                            "vlm_raw_text": raw.vlm_layer_raw_text,
                            "normalized_result": normalized,
                            "vlm_layer_package": layer_pkg.__dict__,
                            "vlm_ack_package": ack.__dict__,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                processed += 1

    elapsed = time.time() - started
    print(f"Processed {processed} tasks in {elapsed:.2f}s ({processed / max(1e-6, elapsed):.2f} tasks/s)")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
