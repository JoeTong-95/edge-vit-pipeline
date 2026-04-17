# Configuration Layer Changes

## 2026-04-16 — Branch `jetson-optimization-vlm-latency`

- Added `config.jetson.vlm-gpu-test.yaml`:
  - Same YOLO TRT FP16 path as `config.jetson.yaml`
  - Forces `config_vlm_device: cuda` to validate VLM-on-GPU viability on Jetson.
- Added `config.jetson.vlm-gpu-lowlat.yaml`:
  - VLM on GPU with latency-first worker tuning (`batch_size=1`, `batch_wait_ms=0`, `crop_cache_size=4`).
  - Intended for A/B latency experiments, not default deployment.
- Updated `README.md`:
  - Documented `config_vlm_device` and `config_yolo_imgsz`.
  - Added a Jetson profile matrix so benchmark commands are reproducible.
