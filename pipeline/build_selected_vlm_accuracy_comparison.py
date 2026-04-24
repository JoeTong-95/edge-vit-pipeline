#!/usr/bin/env python3
"""
Build a first-pass accuracy comparison table from compare_against_human_truth artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "review-package" / "artifacts"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build first-pass selected accuracy comparison from truth artifacts.")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    comparison_paths = sorted(artifacts_dir.glob("compare_against_human_truth_*.json"), key=lambda p: p.name)

    rows: list[dict[str, Any]] = []
    for path in comparison_paths:
        payload = _read_json(path)
        tracker = payload.get("tracker_metrics") or {}
        vlm = payload.get("vlm_metrics") or {}
        metadata = payload.get("metadata_quality_metrics") or {}
        rows.append(
            {
                "artifact": _safe_rel(path),
                "run_id_filter": str(payload.get("run_id_filter", "")),
                "tracker_precision_like": tracker.get("tracker_precision_like", ""),
                "total_human_confirmed_trucks": tracker.get("total_human_confirmed_trucks", ""),
                "vlm_type_agreement_rate": vlm.get("vlm_type_agreement_rate", ""),
                "vlm_class_agreement_rate": vlm.get("vlm_class_agreement_rate", ""),
                "vlm_retry_rate": vlm.get("vlm_retry_rate", ""),
                "problematic_metadata_rate": metadata.get("problematic_metadata_rate", ""),
            }
        )

    payload = {
        "schema": "selected_vlm_accuracy_comparison_v1",
        "created_at_utc": utc_iso(),
        "rows": rows,
        "note": "This table reflects only currently available truth-comparison artifacts; it is expected to grow as more reviewed runs are added.",
    }

    stamp = utc_stamp()
    out_json = artifacts_dir / f"selected_vlm_accuracy_comparison_{stamp}.json"
    out_md = artifacts_dir / f"selected_vlm_accuracy_comparison_{stamp}.md"
    out_csv = artifacts_dir / f"selected_vlm_accuracy_comparison_{stamp}.csv"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Selected VLM Accuracy Comparison",
        "",
        f"- Generated at: `{payload['created_at_utc']}`",
        f"- Artifact count: `{len(rows)}`",
        f"- Note: `{payload['note']}`",
        "",
        "| Run filter | Tracker precision-like | VLM type agreement | VLM class agreement | VLM retry rate | Metadata problematic rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_id_filter']} | {row['tracker_precision_like']} | {row['vlm_type_agreement_rate']} | {row['vlm_class_agreement_rate']} | {row['vlm_retry_rate']} | {row['problematic_metadata_rate']} |"
        )
    if not rows:
        lines.append("| (none) |  |  |  |  |  |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "artifact",
                "run_id_filter",
                "tracker_precision_like",
                "total_human_confirmed_trucks",
                "vlm_type_agreement_rate",
                "vlm_class_agreement_rate",
                "vlm_retry_rate",
                "problematic_metadata_rate",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(json.dumps(payload, indent=2))
    print(f"\nWrote:\n- {out_json}\n- {out_md}\n- {out_csv}")


if __name__ == "__main__":
    main()
