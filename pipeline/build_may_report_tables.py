#!/usr/bin/env python3
"""
Build one consolidated May-report summary from the current review-package artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVIEW_ROOT = REPO_ROOT / "review-package"
DEFAULT_ARTIFACTS_DIR = DEFAULT_REVIEW_ROOT / "artifacts"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _latest_file(artifacts_dir: Path, pattern: str) -> Path | None:
    matches = sorted(artifacts_dir.glob(pattern), key=lambda p: p.name)
    return matches[-1] if matches else None


def _load_latest_matrix(artifacts_dir: Path, include_planned: bool) -> dict[str, Any] | None:
    candidates = sorted(artifacts_dir.glob("vlm_backend_device_matrix_summary_*.json"), key=lambda p: p.name, reverse=True)
    for path in candidates:
        payload = _read_json(path)
        cases = list(payload.get("cases") or [])
        if include_planned:
            return {"path": path, "payload": payload}
        if any(case.get("status") == "ok" for case in cases):
            return {"path": path, "payload": payload}
    return None


def _baseline_status(review_root: Path) -> list[dict[str, Any]]:
    runs_dir = review_root / "runs"
    if not runs_dir.exists():
        return []

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(runs_dir.iterdir(), key=lambda p: p.name):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if "may_baseline_day" not in name and "may_baseline_night" not in name:
            continue
        summary_json = run_dir / "summaries" / "run_summary.json"
        events_jsonl = run_dir / "metadata" / "run_events.jsonl"
        has_terminal_event = False
        if events_jsonl.exists():
            try:
                for line in events_jsonl.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    if json.loads(line).get("event_type") == "run_summary":
                        has_terminal_event = True
                        break
            except Exception:
                has_terminal_event = False

        status = "completed" if summary_json.exists() or has_terminal_event else "in_progress"
        rows.append(
            {
                "run_id": name,
                "label": "day" if "may_baseline_day" in name else "night",
                "status": status,
                "summary_json_exists": summary_json.exists(),
                "terminal_event_exists": has_terminal_event,
                "run_dir": _safe_rel(run_dir),
            }
        )
    return rows


def _extract_speed_rows(matrix_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in matrix_payload.get("cases") or []:
        rows.append(
            {
                "backend": str(case.get("backend", "")),
                "device": str(case.get("device", "")),
                "status": str(case.get("status", "")),
                "avg_infer_ms": case.get("avg_infer_ms", ""),
                "queries_per_sec": case.get("queries_per_sec", ""),
                "selection_reason": str(case.get("selection_reason", "")),
                "reason": str(case.get("reason") or case.get("skip_reason") or case.get("error") or ""),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consolidated May-report tables from current artifacts.")
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--include-planned-matrix", action="store_true")
    args = parser.parse_args()

    review_root = args.review_root.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    baseline_pair_path = _latest_file(artifacts_dir, "report_baseline_pair_*.json")
    truth_path = _latest_file(artifacts_dir, "compare_against_human_truth_*.json")
    matrix_info = _load_latest_matrix(artifacts_dir, include_planned=bool(args.include_planned_matrix))

    baseline_pair = _read_json(baseline_pair_path) if baseline_pair_path else None
    truth = _read_json(truth_path) if truth_path else None
    matrix_payload = matrix_info["payload"] if matrix_info else None
    matrix_path = matrix_info["path"] if matrix_info else None
    baseline_runs = _baseline_status(review_root)
    speed_rows = _extract_speed_rows(matrix_payload) if matrix_payload else []

    payload = {
        "schema": "may_report_tables_v1",
        "created_at_utc": utc_iso(),
        "review_root": str(review_root),
        "artifacts_dir": str(artifacts_dir),
        "sources": {
            "baseline_pair": _safe_rel(baseline_pair_path) if baseline_pair_path else "",
            "matrix_summary": _safe_rel(matrix_path) if matrix_path else "",
            "truth_comparison": _safe_rel(truth_path) if truth_path else "",
        },
        "baseline_pair": baseline_pair,
        "baseline_runs": baseline_runs,
        "speed_comparison_rows": speed_rows,
        "truth_comparison": truth,
    }

    stamp = utc_stamp()
    out_json = artifacts_dir / f"may_report_tables_{stamp}.json"
    out_md = artifacts_dir / f"may_report_tables_{stamp}.md"
    out_speed_csv = artifacts_dir / f"may_report_speed_table_{stamp}.csv"
    out_baseline_csv = artifacts_dir / f"may_report_baseline_status_{stamp}.csv"

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# May Report Tables",
        "",
        f"- Generated at: `{payload['created_at_utc']}`",
        f"- Baseline pair source: `{payload['sources']['baseline_pair'] or 'missing'}`",
        f"- Matrix summary source: `{payload['sources']['matrix_summary'] or 'missing'}`",
        f"- Truth comparison source: `{payload['sources']['truth_comparison'] or 'missing'}`",
        "",
        "## Baseline Runs",
        "",
    ]
    if baseline_runs:
        for row in baseline_runs:
            lines.append(
                f"- `{row['label']}` run `{row['run_id']}`: `{row['status']}` "
                f"(summary_json={row['summary_json_exists']}, terminal_event={row['terminal_event_exists']})"
            )
    else:
        lines.append("- No baseline runs found.")

    lines.extend(["", "## Speed Comparison", ""])
    if speed_rows:
        lines.append("| Backend | Device | Status | Avg infer ms | Q/s | Reason |")
        lines.append("| --- | --- | --- | ---: | ---: | --- |")
        for row in speed_rows:
            avg_ms = "" if row["avg_infer_ms"] in ("", None) else f"{float(row['avg_infer_ms']):.2f}"
            qps = "" if row["queries_per_sec"] in ("", None) else f"{float(row['queries_per_sec']):.4f}"
            reason = (row["reason"] or row["selection_reason"]).replace("|", "/").replace("\n", " ")
            lines.append(f"| {row['backend']} | {row['device']} | {row['status']} | {avg_ms} | {qps} | {reason} |")
    else:
        lines.append("- No matrix summary rows available yet.")

    lines.extend(["", "## Accuracy Snapshot", ""])
    if truth:
        tracker = truth.get("tracker_metrics") or {}
        vlm = truth.get("vlm_metrics") or {}
        metadata = truth.get("metadata_quality_metrics") or {}
        lines.extend(
            [
                f"- Tracker precision-like: `{tracker.get('tracker_precision_like', '')}`",
                f"- Human-confirmed trucks: `{tracker.get('total_human_confirmed_trucks', '')}`",
                f"- VLM type agreement: `{vlm.get('vlm_type_agreement_rate', '')}`",
                f"- VLM class agreement: `{vlm.get('vlm_class_agreement_rate', '')}`",
                f"- Metadata problematic rate: `{metadata.get('problematic_metadata_rate', '')}`",
            ]
        )
    else:
        lines.append("- No truth comparison artifact available yet.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_csv(
        out_speed_csv,
        speed_rows,
        ["backend", "device", "status", "avg_infer_ms", "queries_per_sec", "selection_reason", "reason"],
    )
    _write_csv(
        out_baseline_csv,
        baseline_runs,
        ["run_id", "label", "status", "summary_json_exists", "terminal_event_exists", "run_dir"],
    )

    print(json.dumps(payload, indent=2))
    print(f"\nWrote:\n- {out_json}\n- {out_md}\n- {out_speed_csv}\n- {out_baseline_csv}")


if __name__ == "__main__":
    main()
