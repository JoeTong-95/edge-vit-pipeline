# Jetson Optimization Plan — YOLO Latency (Primary Bottleneck)
**Branch**: `jetson-optimization-yolo-latency`  
**Date**: 2026-04-16  
**Baseline config**: `src/configuration-layer/config.jetson.yaml`  

## 1) Problem statement
Fresh benchmark on `jetson-dev` shows end-to-end FPS is now limited primarily by YOLO inference latency.

### Measured baseline (latest run)
- `estimated_fps`: **20.43**
- `avg_end_to_end_ms`: **48.96 ms**
- `avg_yolo_ms`: **39.48 ms**  ← dominant cost
- `avg_input_ms`: 8.50 ms
- `avg_tracking_ms`: 0.59 ms
- `avg_state_ms`: 0.09 ms
- `avg_vlm_ms` (main thread async submit): 0.07 ms
- YOLO capacity (`total_yolo_s / frames`): **25.33 fps**

Interpretation: YOLO consumes ~80% of the frame budget; improving other layers gives small gains unless YOLO latency is reduced.

---

## 2) Why YOLO is still expensive after TRT FP16
Current engine is fixed-shape TRT FP16 at 640x640. Even when ROI crops are tiny, TRT engine still runs full 640x640 compute per frame.

Likely contributors:
1. Fixed 640x640 TRT shape ignores ROI compute reduction.
2. Ultralytics TensorRT wrapper may add per-call Python overhead.
3. Engine built for max quality (larger model) instead of lower latency target.

---

## 3) Optimization hypothesis
Biggest win now is reducing YOLO per-frame compute and invocation overhead.

### Primary hypothesis
**Move from fixed 640x640 path to a latency-oriented YOLO inference path**:
- FP16 TRT engine at smaller effective shape (e.g., 512 or 384x640 FP16), OR
- Dynamic-shape TRT engine/profile (if stable on Jetson), OR
- Direct TensorRT runtime path (bypass Ultralytics wrapper in hot loop).

Expected gain target: `avg_yolo_ms` from ~39.5 ms -> <=28 ms (pipeline >=28 fps).

---

## 4) Experiments (ordered)

### Experiment A — TRT FP16 resolution sweep (lowest risk)
Build and benchmark FP16 engines at:
- 640x640 (control)
- 512x512
- 384x640

For each:
- run 3 benchmark passes
- compare `avg_yolo_ms`, `estimated_fps`, and detection quality proxy (`avg dets/frame`)

Decision rule:
- choose fastest engine with acceptable detection retention (no major collapse in truck detections/tracks).

### Experiment B — Direct TRT runtime benchmark (medium risk)
Prototype direct TensorRT execution (no Ultralytics in-frame call path) and compare per-frame latency against current `run_yolo_detection`.

Decision rule:
- adopt only if >=10% latency drop without regression risk in outputs.

### Experiment C — Dynamic-shape TRT profile (higher risk)
Try min/opt/max shape profile tied to ROI behavior.
- if Jetson/TRT stability issues (allocator or profile mismatch), stop and revert.

Decision rule:
- keep only if stable and faster than best static FP16 engine.

### Experiment D — Input path micro-optimizations (secondary)
If YOLO gets near target and FPS still <30, optimize input decode/resize path (~8.5 ms currently):
- reduce color/resize copies
- validate frame decode threading

---

## 5) Non-goals for this branch
- VLM speed optimization (async worker not frame-critical here)
- Tracking/state refactors (currently low ms impact)
- Docker path resurrection (native Jetson path remains canonical)

---

## 6) Deliverables for this branch
1. Bench report table (per experiment, 3-run average)
2. Selected YOLO runtime path + config updates
3. Updated `JETSON_OPTIMIZATION.md` section with results and rationale
4. Repro commands for chosen engine build + benchmark

---

## 7) Success criteria
- Primary: `estimated_fps >= 28` on `config.jetson.yaml`
- Secondary: YOLO capacity >= 30 fps sustained
- No regression in pipeline stability (no allocator asserts, no runtime abort at teardown)
