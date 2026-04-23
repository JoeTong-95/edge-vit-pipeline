# Review Package

This top-level folder is for locally generated May-report review runs.

Expected generated layout:

- `runs/<run_id>/new_tracks/`
- `runs/<run_id>/vlm_accepted_targets/`
- `runs/<run_id>/metadata/`
- `runs/<run_id>/summaries/`
- `runs/<run_id>/artifacts/`
- `human_truth.sqlite`

This folder is intentionally repo-visible so teammates can SSH in and review outputs on the Jetson.

Implementation details and exact contracts are defined in:

- `may-report-package/review_package_spec.md`
