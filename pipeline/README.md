This folder defines the locked pipeline contracts and ownership boundaries for the repo.

## Key Files

- `pipeline_layers_and_interactions.md`: the main source of truth for layer names, public functions, package shapes, and interactions.
- `codex_ground_rules.md`: repo-specific working constraints, including the rule to stay inside one layer unless explicitly asked.
- `master-pipeline.png`: visual reference for the pipeline structure.

## Current Practical Flow

The currently runnable path in this branch is:

`configuration_layer -> input_layer -> yolo_layer -> tracking_layer`

The contract still comes from `pipeline_layers_and_interactions.md`, even when local helper scripts are used for visualization or evaluation.

## Current Utility Scripts Added Around The Contract

These scripts are helpers around the layer APIs, not replacements for the layer contract itself:

- `src/yolo-layer/visualize_yolo.py`: config-driven YOLO-only visualization with optional live preview and SQLite metrics.
- `src/tracking-layer/visualize_tracking.py`: config-driven YOLO plus tracking visualization with optional live preview and SQLite metrics.
- `src/tracking-layer/automated_evaluation.py`: sequential benchmark sweep across CPU/CUDA, YOLO v8/v10/v11, and tracking on/off.
- `src/tracking-layer/plot_evaluation_results.py`: creates styled summary plots from the evaluation SQLite output.

## How To Use This Folder

1. Start with `pipeline_layers_and_interactions.md` when deciding whether a layer interface is correct.
2. Use the layer-local README files for runnable examples and practical commands.
3. Treat helper scripts as orchestration utilities that must stay compatible with the pipeline contract.
