# Pipeline Documentation Changes

## 2026-04-16 — Branch `jetson-optimization-vlm-latency`

- No pipeline contract shape changes yet.
- Performance investigation branch created to optimize deployment-limiting VLM latency on Jetson.
- Active tracking document for this branch is:
  - `branch-optimizer-log.md`
- Config-layer documentation was updated with Jetson profile usage and benchmark commands:
  - `src/configuration-layer/README.md`
- VLM semantic contract trimmed by removing `estimated_weight_kg` while preserving
  end-to-end package flow and acknowledgement semantics (`is_truck`,
  `wheel_count`, `ack_status`, `retry_reasons` remain).
