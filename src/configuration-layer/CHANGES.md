# Configuration Layer Changes

## 2026-04-15

- Added VLM runtime knobs to the shared config contract: `config_vlm_runtime_mode`, `config_vlm_worker_max_queue_size`, `config_vlm_worker_batch_size`, `config_vlm_worker_batch_wait_ms`, `config_vlm_worker_spill_queue_path`, `config_vlm_spill_max_file_mb`, and `config_vlm_realtime_throttle_enabled`.
- Updated defaults / schema / types / normalizer / validator so these knobs are available everywhere that reads `config.yaml`.
- Updated `config.yaml` and `README.md` to document and expose the new knobs for benchmark + realtime VLM tools.
- `config_vlm_spill_max_file_mb` caps the active spill JSONL size by rotating to a sibling `*.jsonl.rotated.<ms>` file (delete or archive rotated files separately for 24/7 deployments).

## 2026-04-10

- Added `config_vlm_dead_after_lost_frames` to the shared configuration contract so the cropper can declare a track dead after a configurable number of consecutive `lost` tracking updates.
- Updated `config_defaults.py`, `config_schema.py`, `config_types.py`, and `config_normalizer.py` so the new dead-threshold setting is normalized and carried like the rest of the pipeline config.
- Updated `config_validator.py` so `config_vlm_dead_after_lost_frames` must be greater than 0.
- Updated `config.yaml` to expose the dead-threshold control for the cropper and VLM visualizers.
- Updated `README.md` to document the new dead-threshold setting and how it affects partial-cache finalization.

## 2026-04-08

- Updated `config.yaml` to point to the existing `data/sample.mp4` file instead of the missing `data/sample1.mp4`.
- Updated the default `config_input_path` in `config_defaults.py` to `data/sample.mp4` so a default video-mode configuration is runnable from the repo.
- Updated the default `config_yolo_model` in `config_defaults.py` and `config.yaml` to `yolov8n.pt` to match the bundled local YOLO weights.
- Tightened `validate_config` via `config_validator.py` so video-mode configs fail fast when `config_input_path` does not exist.
- Updated `config_validator.py` to resolve relative `config_input_path` values against the repo root as well, so scripts launched from layer subfolders still honor the shared config file.
- Rewrote README.md to document the current config-driven workflow, including config.yaml, accepted config_device values, bundled YOLO model names, and the scripts that now consume this layer by default.

## 2026-04-09

- Added `config_vlm_crop_cache_size` to the shared configuration contract so the cropper-side local cache size can be controlled from `config.yaml`.
- Updated `config_defaults.py`, `config_schema.py`, `config_types.py`, and `config_normalizer.py` so the new cache-size setting is normalized like the rest of the pipeline config.
- Updated `config_validator.py` so `config_vlm_crop_cache_size` must be greater than 0.
- Updated `config.yaml` to expose the new cache-size control for the cropper and cropper visualizer.
- Updated the shared visualizer defaults so ROI, YOLO, tracking, and cropper helpers no longer auto-save videos into `data/` unless `--output` is passed explicitly.
- Documented that `config_vlm_crop_cache_size` now directly controls the one-shot VLM candidate collection window before dispatch.

## 2026-04-09 (documentation and defaults)

- Updated `config.yaml` so `config_vlm_model` points at the bundled `src/vlm-layer/Qwen3.5-0.8B` directory and `config_vlm_enabled` reflects the VLM path used by `visualize_vlm.py` (adjust per machine as needed).
- Refreshed `README.md` with VLM-related keys, relative links, and consumers including `visualize_vlm.py`.
- Added `config_vlm_crop_feedback_enabled` so the optional VLM path can run either as a true feedback loop (`true`) or as single-shot classify-and-complete (`false`).
