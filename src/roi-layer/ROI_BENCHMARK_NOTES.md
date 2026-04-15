## ROI vs YOLO performance notes

This note documents why ROI sometimes appears to provide little (or no) YOLO speedup in the end-to-end benchmark output, even when ROI bounds are much smaller than the full frame.

### Summary

- **ROI cropping works**: it reduces the image pixels fed into YOLO after ROI locks.
- **YOLO inference tends to get faster with smaller ROI** (measured in isolation).
- In the end-to-end benchmark, the printed `avg_yolo_ms` originally bundled **inference + Python post-processing**, which dilutes the measured speedup.
- ROI comparisons can also be misleading if ROI locks late and only a small number of post-lock frames were measured.

### Evidence: isolated YOLO-only ROI study

Script: `src/evaluation-output-layer/roi_study.py`

This study isolates **Layer 3 ROI + Layer 4 YOLO** and measures:

- YOLO on full frames
- YOLO on ROI-cropped frames (after ROI locks)

On the included sample videos (`data/sample1..4.mp4`), ROI reduced average pixels and improved YOLO mean inference time by roughly:

- sample1: ~**1.07x**
- sample2: ~**1.23x**
- sample3: ~**1.22x**
- sample4: ~**1.24x**

(Exact values depend on hardware, drivers, and model version.)

### Why the end-to-end benchmark sometimes shows ~1.0x

1) **Timing block included post-processing**

The benchmark’s YOLO timing was originally:

- `run_yolo_detection(...)` (model inference)
- `filter_yolo_detections(...)` (Python filtering)
- `build_yolo_layer_package(...)` (Python packaging)

The latter two are mostly **fixed overhead** that does not shrink proportionally with ROI pixels. If inference is already relatively fast, that constant overhead can dominate, making ROI speedups look small.

To address this, `benchmark.py` now prints:

- `yolo_infer_ms` (inference only)
- `yolo_post_ms` (filter + package)

2) **Undersampled post-lock window**

If ROI locks late, a short measurement window may include only a small number of post-lock frames, so the post-lock average is noisy.

To address this, `benchmark.py` supports a benchmark-only mode:

- `MEASURE_ONLY_AFTER_ROI_LOCK = True`

This does not change runtime ROI behavior; it only ensures benchmarking compares ROI performance with enough post-lock frames.

### Recommendation

- ROI is **worth keeping** if it consistently locks to a meaningfully smaller area (e.g. < ~50% of frame pixels).
- When evaluating ROI performance, rely on **`yolo_infer_ms`** more than the combined `avg_yolo_ms`.
- Use `MEASURE_ONLY_AFTER_ROI_LOCK=True` for ROI comparisons to avoid misleading small-sample results.

