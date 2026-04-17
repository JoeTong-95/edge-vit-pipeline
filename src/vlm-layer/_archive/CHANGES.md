# VLM Layer Changes

## 2026-04-16 — Branch `jetson-optimization-vlm-latency`

- Removed `estimated_weight_kg` from the active VLM semantic contract:
  - prompt schema now asks for `is_truck`, `wheel_count`, `ack_status`, `retry_reasons`
  - parser/normalizer no longer emit or expect `estimated_weight_kg`
  - sample/debug JSON payloads updated accordingly
- Reduced generation budget from `max_new_tokens=64` to `max_new_tokens=32` to fit the slimmer JSON schema while preserving full pipeline semantics.
- Updated visual debug JSON card (`visualize_vlm.py`) to display only active contract fields.

## 2026-04-16 — VLM inference latency (SmolVLM / Jetson)

- `VLMConfig.config_vlm_max_new_tokens` + `VLMRuntimeState.vlm_runtime_max_new_tokens`: generation cap is now **config-driven** (`config_vlm_max_new_tokens` in YAML, default **32**). Jetson SmolVLM profile sets **24** for a shorter decode tail on the small JSON contract.
- On **CUDA**, model load tries **`attn_implementation="sdpa"`** first (faster attention when Transformers supports it for the checkpoint), then falls back to the default path.
- `initialize_pipeline.py` now respects **`config_vlm_device`** when initializing the VLM (same behaviour as `benchmark.py`).
