#!/usr/bin/env python3
"""
roi_benchmark_matrix.py

Runs `pipeline/benchmark.py` across multiple videos with ROI ON vs OFF,
without editing config.yaml. This is for investigation and reporting only.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BENCH_PATH = REPO_ROOT / "pipeline" / "benchmark.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("edge_benchmark", BENCH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark module from: {BENCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    videos = [REPO_ROOT / "data" / f"sample{i}.mp4" for i in (1, 2, 3, 4)]
    videos = [v for v in videos if v.is_file()]
    if not videos:
        raise RuntimeError("No sample*.mp4 videos found under data/.")

    # Keep these short-ish; we want relative comparisons, not full 60s runs.
    warmup_s = 2.0
    measure_s = 8.0

    bench = _load_benchmark_module()

    for video in videos:
        for roi_enabled in (False, True):
            print()
            print("=" * 88)
            print(f"video={video.name} | roi_enabled={roi_enabled}")
            print("=" * 88)

            bench.WARMUP_SECONDS = warmup_s
            bench.MEASURE_SECONDS = measure_s
            bench.MEASURE_ONLY_AFTER_ROI_LOCK = True
            bench.BENCH_OVERRIDE_ROI_ENABLED = roi_enabled
            bench.BENCH_OVERRIDE_INPUT_PATH = f"data/{video.name}"

            bench.main()

    print()
    print("Done. Review YOLO pre/post ROI frames_measured + avg_pixels for each run.")


if __name__ == "__main__":
    main()

