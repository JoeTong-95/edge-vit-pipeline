#!/usr/bin/env python3
"""
Build a first-pass speed comparison for the currently selected realistic VLM configs.
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest(path_dir: Path, pattern: str) -> Path | None:
    matches = sorted(path_dir.glob(pattern), key=lambda p: p.name)
    return matches[-1] if matches else None


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _lookup_case(cases: list[dict[str, Any]], backend: str, device: str) -> dict[str, Any] | None:
    for case in cases:
        if str(case.get("backend")) == backend and str(case.get("device")) == device:
            return case
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build first-pass speed comparison for selected realistic VLM configs.")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    planned_path = _latest(artifacts_dir, "vlm_backend_device_matrix_summary_*2439.json")
    if planned_path is None:
        # Fallback to latest planned matrix summary if the exact timestamp suffix is absent.
        candidates = sorted(artifacts_dir.glob("vlm_backend_device_matrix_summary_*.json"), key=lambda p: p.name, reverse=True)
        planned_path = None
        for path in candidates:
            payload = _read_json(path)
            if payload.get("dry_run") and str(payload.get("preset")) == "may_report_realistic":
                planned_path = path
                break
    measured_path = _latest(artifacts_dir, "vlm_backend_device_matrix_summary_20260424_000432.json")
    gemma_path = _latest(artifacts_dir, "vlm_backend_device_matrix_summary_20260424_000632.json")

    if planned_path is None:
        raise FileNotFoundError("Could not locate a realistic-preset dry-run matrix artifact.")
    if measured_path is None:
        raise FileNotFoundError("Could not locate the measured smol matrix artifact.")
    if gemma_path is None:
        raise FileNotFoundError("Could not locate the gemma follow-up matrix artifact.")

    planned = _read_json(planned_path)
    measured = _read_json(measured_path)
    gemma = _read_json(gemma_path)

    planned_cases = list(planned.get("cases") or [])
    measured_cases = list(measured.get("cases") or [])
    gemma_cases = list(gemma.get("cases") or [])

    rows: list[dict[str, Any]] = []
    for planned_case in planned_cases:
        backend = str(planned_case.get("backend", ""))
        device = str(planned_case.get("device", ""))
        selection_reason = str(planned_case.get("selection_reason", ""))

        source_case = _lookup_case(gemma_cases, backend, device) or _lookup_case(measured_cases, backend, device) or {}
        rows.append(
            {
                "backend": backend,
                "device": device,
                "selection_reason": selection_reason,
                "status": str(source_case.get("status", "planned")),
                "avg_infer_ms": source_case.get("avg_infer_ms", ""),
                "queries_per_sec": source_case.get("queries_per_sec", ""),
                "init_s": source_case.get("init_s", ""),
                "reason": str(source_case.get("reason") or source_case.get("skip_reason") or source_case.get("error") or ""),
            }
        )

    smol_cpu = _lookup_case(rows, "smolvlm_256m", "cpu")
    smol_cuda = _lookup_case(rows, "smolvlm_256m", "cuda")
    recommendation = ""
    if smol_cpu and smol_cuda and smol_cpu.get("avg_infer_ms") and smol_cuda.get("avg_infer_ms"):
        cpu_ms = float(smol_cpu["avg_infer_ms"])
        cuda_ms = float(smol_cuda["avg_infer_ms"])
        speedup = cpu_ms / cuda_ms if cuda_ms else 0.0
        recommendation = (
            f"Use smolvlm_256m on cuda as the first measured performance reference on this Jetson; "
            f"the current single-image benchmark shows about {speedup:.1f}x lower inference latency than cpu."
        )

    payload = {
        "schema": "selected_vlm_speed_comparison_v1",
        "created_at_utc": utc_iso(),
        "sources": {
            "planned_matrix": _safe_rel(planned_path),
            "measured_matrix": _safe_rel(measured_path),
            "gemma_followup_matrix": _safe_rel(gemma_path),
        },
        "rows": rows,
        "recommendation": recommendation,
    }

    stamp = utc_stamp()
    out_json = artifacts_dir / f"selected_vlm_speed_comparison_{stamp}.json"
    out_md = artifacts_dir / f"selected_vlm_speed_comparison_{stamp}.md"
    out_csv = artifacts_dir / f"selected_vlm_speed_comparison_{stamp}.csv"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Selected VLM Speed Comparison",
        "",
        f"- Generated at: `{payload['created_at_utc']}`",
        f"- Planned matrix: `{payload['sources']['planned_matrix']}`",
        f"- Measured matrix: `{payload['sources']['measured_matrix']}`",
        f"- Gemma follow-up: `{payload['sources']['gemma_followup_matrix']}`",
        "",
        "| Backend | Device | Status | Avg infer ms | Q/s | Reason |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        avg_ms = "" if row["avg_infer_ms"] in ("", None) else f"{float(row['avg_infer_ms']):.2f}"
        qps = "" if row["queries_per_sec"] in ("", None) else f"{float(row['queries_per_sec']):.4f}"
        reason = (row["reason"] or row["selection_reason"]).replace("|", "/").replace("\n", " ")
        lines.append(f"| {row['backend']} | {row['device']} | {row['status']} | {avg_ms} | {qps} | {reason} |")
    if recommendation:
        lines.extend(["", "## Recommendation", "", f"- {recommendation}"])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["backend", "device", "selection_reason", "status", "avg_infer_ms", "queries_per_sec", "init_s", "reason"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(json.dumps(payload, indent=2))
    print(f"\nWrote:\n- {out_json}\n- {out_md}\n- {out_csv}")


if __name__ == "__main__":
    main()
