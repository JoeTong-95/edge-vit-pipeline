Layer changes in this branch

- ## 2026-04-15
-
- Added `ROI_BENCHMARK_NOTES.md` documenting why ROI sometimes appears to provide little YOLO speedup in end-to-end runs, and how to benchmark ROI correctly (post-lock sampling + infer vs post-processing timing).

- Added `roi_layer.py` implementing the pipeline contract public API:
  `initialize_roi_layer`, `update_roi_state`, `apply_roi_to_frame`, and
  `build_roi_layer_package`.
- Implemented the documented ROI node helpers for candidate-box collection,
  ROI-bound computation, bound locking, cropping, and passthrough behavior.
- Added integration compatibility so ROI accepts both dict-style input fixtures
  and the real `InputLayerPackage` dataclass emitted by the input layer on the
  confirmed main path.
- Added `test_roi_layer.py` covering disabled passthrough, startup discovery,
  ROI lock, cropped output, and dataclass-package compatibility.
- Added `visualize_roi.py` to render ROI discovery, ROI lock state, and a live
  side-by-side view of the full frame and active ROI crop.
- Updated `visualize_roi.py` so it only writes a video when `--output` is provided, instead of auto-saving into `data/` on every run.

## 2026-04-09

- Updated `README.md` with `visualize_roi.py` quick start and a pointer to `pipeline/README.md` for the full helper list.
- Changed ROI discovery counting so overlapping repeat detections are deduplicated inside the ROI layer before thresholding; this makes `config_roi_vehicle_count_threshold` behave closer to a unique startup-vehicle count.

## 2026-04-10

- Added `visualize_roi_vlm.py`, a cross-layer helper that shows ROI calibration
  first and then, after ROI lock, runs YOLO and VLM only inside the cropped ROI
  region.
- Updated `README.md` to document the new ROI -> VLM helper and its phase
  change from calibration to ROI-only downstream inference.
- Added `benchmark_roi_yolo.py`, a batch ROI-vs-full-frame benchmark that runs
  sample videos 1-4, logs per-run and per-frame performance into SQLite, and
  writes a styled ROI comparison chart.
- Split visual GIF rendering out of the benchmark path into
  `render_roi_yolo_gif.py`, so benchmarking stays headless and does not spend
  time on annotation or GIF export inside the measured loop.
- Refined `benchmark_roi_yolo.py` so the comparison now matches the explicit
  evaluation request: direct full-video YOLO input versus a separately
  calibrated fixed ROI crop used as the YOLO input.
- Added a `--roi-threshold` override to `benchmark_roi_yolo.py` so ROI
  calibration sweeps can be run without editing `config.yaml`.
- Updated `README.md` with the new headless ROI benchmark behavior and the
  separate ROI pipeline GIF renderer.
- Added `save_roi_videos.py`, a batch helper that saves the `visualize_roi.py`
  style ROI calibration/crop videos for sample1-4 into the desktop ROI output
  folder.
