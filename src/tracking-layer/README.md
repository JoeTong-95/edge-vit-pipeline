# Tracking Layer

This layer owns object persistence and lifecycle labeling on top of YOLO detections.

## Main Files

- `tracker.py`: tracking layer public API and ByteTrack-backed state
- `visualize_tracking.py`: config-driven live or offline visualization for YOLO plus tracking
- `automated_evaluation.py`: sequential evaluation sweep across model, device, and tracking mode
- `plot_evaluation_results.py`: plotting utility for the evaluation SQLite output
- `test_tracking_layer.py`: layer-local validation script

## Public API

The pipeline contract for this layer is implemented in `tracker.py`:
- `initialize_tracking_layer`
- `update_tracks`
- `assign_tracking_status`
- `build_tracking_layer_package`

The emitted package matches the documented flat `tracking_layer_*` field layout.

## Quick Start

From `src/tracking-layer`:

```powershell
python .\visualize_tracking.py
python .\visualize_tracking.py --show
python .\visualize_tracking.py --save-metrics
```

The visualizer reads defaults from `src/configuration-layer/config.yaml` for:
- input video path
- YOLO model
- confidence threshold
- device mode

## Automated Evaluation Sweep

Run the current benchmark sweep with:

```powershell
python .\automated_evaluation.py --max-seconds 60
```

Current sweep combinations:
- device: `cpu`, `cuda`
- model family: `v8`, `v10`, `v11`
- mode: detection only, tracking enabled

Outputs are saved by default to a configurable directory (see `automated_evaluation.py` defaults; override with CLI flags):
- one MP4 per run, such as `eval_v8_cuda_tracking.mp4`
- one SQLite database with run-level and frame-level metrics

## Plotting Evaluation Results

After the sweep, generate a summary plot with:

```powershell
python .\plot_evaluation_results.py
```

The plot utility reads the newest `tracking_eval_metrics*.sqlite` file from the evaluator output directory by default and saves a styled PNG summary next to it.

## Notes

- The tracking visualizer HUD shows mode, end-to-end FPS, inference-only FPS, and track counts.
- SQLite logging is built into both the visualizer and automated evaluator.
- The evaluator handles older SQLite schemas and falls back to a new DB file if the default one is still locked by an older run.
- The end-to-end VLM demo `src/vlm-layer/visualize_vlm.py` also consumes this layer’s package shape via the shared `tracker.py` API (see `pipeline/README.md` for helper scope).
