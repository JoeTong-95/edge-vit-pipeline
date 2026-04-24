#!/usr/bin/env python3
"""
Build a simple index of the latest May-report artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "review-package" / "artifacts"


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _latest(artifacts_dir: Path, pattern: str) -> str:
    matches = sorted(artifacts_dir.glob(pattern), key=lambda p: p.name)
    return _safe_rel(matches[-1]) if matches else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an index of the latest May-report artifacts.")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema": "may_report_artifact_index_v1",
        "created_at_utc": utc_iso(),
        "artifacts": {
            "baseline_pair": _latest(artifacts_dir, "report_baseline_pair_*.json"),
            "may_report_tables": _latest(artifacts_dir, "may_report_tables_*.json"),
            "baseline_status_table": _latest(artifacts_dir, "may_report_baseline_status_*.csv"),
            "speed_table": _latest(artifacts_dir, "may_report_speed_table_*.csv"),
            "selected_speed_comparison": _latest(artifacts_dir, "selected_vlm_speed_comparison_*.json"),
            "selected_accuracy_comparison": _latest(artifacts_dir, "selected_vlm_accuracy_comparison_*.json"),
            "matrix_report_summary": _latest(artifacts_dir, "report_summary_*.json"),
            "truth_comparison": _latest(artifacts_dir, "compare_against_human_truth_*.json"),
        },
    }

    stamp = utc_stamp()
    out_json = artifacts_dir / f"may_report_artifact_index_{stamp}.json"
    out_md = artifacts_dir / f"may_report_artifact_index_{stamp}.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# May Report Artifact Index",
        "",
        f"- Generated at: `{payload['created_at_utc']}`",
        "",
    ]
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value or 'missing'}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote:\n- {out_json}\n- {out_md}")


if __name__ == "__main__":
    main()
