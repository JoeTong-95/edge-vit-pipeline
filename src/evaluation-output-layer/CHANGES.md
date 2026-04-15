# Evaluation Output Layer — Changes

## 2026-04-15

- Fixed `emit_evaluation_output(...)` to support `output_destination="stdout"` in addition to the SQLite sink.
  - **Problem**: the end-to-end burner smoke test prints evaluation output to stdout and was failing with `Unsupported output_destination: 'stdout'`.
  - **Fix**: allow `"stdout"` and print a deterministic one-line JSON record when selected.

