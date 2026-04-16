# VLM Layer Changes

## 2026-04-16 — Branch `jetson-optimization-vlm-latency`

- Removed `estimated_weight_kg` from the active VLM semantic contract:
  - prompt schema now asks for `is_truck`, `wheel_count`, `ack_status`, `retry_reasons`
  - parser/normalizer no longer emit or expect `estimated_weight_kg`
  - sample/debug JSON payloads updated accordingly
- Reduced generation budget from `max_new_tokens=64` to `max_new_tokens=32` to fit the slimmer JSON schema while preserving full pipeline semantics.
- Updated visual debug JSON card (`visualize_vlm.py`) to display only active contract fields.
