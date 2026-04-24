# Human Truth Package Gap Note

Date: `2026-04-24`

## Summary

The repo already has the main human-truth package pieces:

- `pipeline/run_deployment_review.py`
- `pipeline/review_app.py`
- `pipeline/compare_against_human_truth.py`
- `review-package/human_truth.sqlite`

But the implementation was not yet fully aligned with the report-package contract.

## Main Gaps Found

1. [x] `vlm_accepted_targets` was not constrained to the report vehicle scope.
   - The report package expects downstream review/export to focus on:
     - `pickup`
     - `van`
     - `truck`
     - `bus`
   - Existing saved review artifacts still included `car` rows.

2. [x] The review app did not expose row-level metadata.
   - It stored only `metadata_relpath` to the CSV file.
   - The UI showed wrapper fields such as `run_id`, `track_id`, and `image_relpath`, but not the actual aligned tracker/VLM row fields that reviewers are supposed to inspect and highlight.

3. [x] Queue behavior hid items after any single label.
   - A single saved label caused an item to disappear from the “next unlabeled” queue.
   - This made the workflow poor for multi-axis review on the same item, especially when a reviewer wants to add:
     - one class decision
     - plus `repeat`
     - plus `bad_crop`
     - plus metadata highlights

4. [x] The comparison script could produce misleading class metrics.
   - The current VLM contract exposes `is_truck` / `is_type`, not a full normalized downstream class label from VLM.
   - Existing class-agreement output was effectively comparing human `true_class` against the saved detector `target_class` proxy.

5. [ ] The package still lacks report-scale human labeling coverage.
   - This is now the main remaining gap.
   - It cannot be completed by code alone.

6. [ ] Multi-config review-package generation still depends on VLM switcher reliability.
   - The human-truth package shape is now largely aligned with the report workflow.
   - But generating review packages for multiple comparison configs still requires:
     - config-driven VLM backend selection
     - config-driven VLM device selection
     - config-driven VLM runtime selection
     - stable end-to-end review-package execution for those config combinations
   - Important Jetson safety note:
     - previous sessions that attempted to smoke-run the VLM switcher / unstable backend-device combinations were associated with SSH connection loss
     - related docs should be re-read before touching that path:
     - `may-report-package/may_report_todo_v2.md`
     - `may-report-package/review_package_spec.md`
     - any future risky switcher rerun should be treated as an explicit operator-confirmed action, not something to launch casually during an exploratory coding pass
   - Safe validation completed in this cycle:
     - added `pipeline/validate_vlm_switcher.py`
     - rechecked all in-repo YAML configs without launching inference
     - `config.report-baseline.yaml` resolves to:
       - backend `smolvlm_256m`
       - device `cpu`
       - runtime `async`
     - `config.yaml` resolves to:
       - backend `smolvlm_256m`
       - device `cuda`
       - runtime `async`
     - `config.jetson.yaml` resolves to:
       - backend `smolvlm_256m`
       - device `cuda`
       - runtime `async`
     - `config.cpu-test.yaml` correctly resolves as VLM-disabled
   - Current conclusion:
     - config parsing and backend/device/runtime resolution are now validated
     - bounded helper-path live validation for `smolvlm_256m` has passed on both `cpu` and `cuda`
     - the remaining switcher risk is live inference stability on selected backend/device combinations in the full review-package path
   - Current practical reading:
     - the `smolvlm_256m` switcher itself is revalidated for bounded helper workloads on `cpu` and `cuda`
     - the already-proven May-report package-generation baseline should still remain the `cpu` VLM path until full `YOLO cuda + VLM cuda` review runs are shown stable

## Fixes Completed In This Cycle

1. Review-app ingestion now filters `vlm_accepted_target` rows to the report vehicle scope only.
2. Review-app ingestion now stores row-level metadata JSON for each `review_item`.
3. Review-app UI now renders row-level metadata fields for actual per-item inspection and highlight capture.
4. Review queue logic now treats classification labels as the completion gate instead of any label at all.
5. Review label saves now keep the current item loaded so multiple labels/highlights can be added before moving on.
6. Review artifact generation now skips out-of-scope VLM accepted-target exports.
7. Truth comparison now filters out-of-scope VLM rows and labels class-agreement output honestly as a detector-class proxy.
8. Review-app ingestion now deletes stale out-of-scope `vlm_accepted_target` rows from `human_truth.sqlite`.

## Remaining Gap After This Fix Pass

Two real gaps remain before the full report workflow is complete:

1. VLM switcher validation for config-specific review-package generation.
2. Real human labeling volume on the generated day/night review packages.

At the time of review:

- `review_items`: sparse
- `review_labels`: sparse
- `review_highlights`: sparse

So the workflow shape is present, but:

- multi-config package generation still depends on a stable VLM switcher path
- generated review packages still need to be reviewed by a human operator to produce meaningful report metrics
