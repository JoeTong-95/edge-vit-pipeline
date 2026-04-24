#!/usr/bin/env python3
"""
Build report-ready summaries from a matrix runner JSON artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _format_ms(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _load_matrix(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": case.get("backend", ""),
        "device": case.get("device", ""),
        "status": case.get("status", ""),
        "selection_reason": case.get("selection_reason", ""),
        "avg_infer_ms": case.get("avg_infer_ms", ""),
        "queries_per_sec": case.get("queries_per_sec", ""),
        "init_s": case.get("init_s", ""),
        "runtime_device": case.get("runtime_device", ""),
        "runtime_dtype": case.get("runtime_dtype", ""),
        "returncode": case.get("returncode", ""),
        "reason": case.get("reason") or case.get("skip_reason") or case.get("error") or "",
        "case_output_json": case.get("case_output_json", ""),
    }


def _compute_summary(matrix: dict[str, Any]) -> dict[str, Any]:
    cases = list(matrix.get("cases") or [])
    ok_cases = [case for case in cases if case.get("status") == "ok"]
    skipped_cases = [case for case in cases if case.get("status") == "skipped"]
    error_cases = [case for case in cases if case.get("status") == "error"]
    planned_cases = [case for case in cases if case.get("status") == "planned"]

    ok_lookup = {
        (str(case.get("backend")), str(case.get("device"))): case
        for case in ok_cases
    }
    backend_ratios: list[dict[str, Any]] = []
    for backend in sorted({str(case.get("backend")) for case in ok_cases}):
        cpu_case = ok_lookup.get((backend, "cpu"))
        cuda_case = ok_lookup.get((backend, "cuda"))
        if not cpu_case or not cuda_case:
            continue
        cpu_ms = float(cpu_case.get("avg_infer_ms", 0.0) or 0.0)
        cuda_ms = float(cuda_case.get("avg_infer_ms", 0.0) or 0.0)
        backend_ratios.append(
            {
                "backend": backend,
                "cpu_avg_infer_ms": cpu_ms,
                "cuda_avg_infer_ms": cuda_ms,
                "cuda_speedup_vs_cpu": round(_safe_ratio(cpu_ms, cuda_ms), 4) if cuda_ms else 0.0,
            }
        )

    return {
        "schema": "vlm_report_summary_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "matrix_source": matrix.get("matrix_source") or matrix.get("source_json") or "",
        "counts": {
            "total_cases": len(cases),
            "ok_cases": len(ok_cases),
            "skipped_cases": len(skipped_cases),
            "error_cases": len(error_cases),
            "planned_cases": len(planned_cases),
        },
        "backend_ratios": backend_ratios,
        "cases": [_flatten_case(case) for case in cases],
    }


def _build_markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# Report Summary",
        "",
        f"- Generated at: `{summary['created_at_utc']}`",
        f"- Total cases: `{counts['total_cases']}`",
        f"- OK: `{counts['ok_cases']}`",
        f"- Skipped: `{counts['skipped_cases']}`",
        f"- Error: `{counts['error_cases']}`",
        f"- Planned: `{counts['planned_cases']}`",
        "",
        "## Case Table",
        "",
        "| Backend | Device | Status | Avg infer ms | Q/s | Reason | Selection reason |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for case in summary["cases"]:
        lines.append(
            "| {backend} | {device} | {status} | {avg_ms} | {qps} | {reason} | {selection_reason} |".format(
                backend=case["backend"],
                device=case["device"],
                status=case["status"],
                avg_ms=_format_ms(case["avg_infer_ms"]),
                qps=_format_ms(case["queries_per_sec"]),
                reason=str(case["reason"]).replace("\n", " ").replace("|", "/"),
                selection_reason=str(case["selection_reason"]).replace("\n", " ").replace("|", "/"),
            )
        )
    if summary["backend_ratios"]:
        lines.extend(
            [
                "",
                "## CPU Vs CUDA Ratios",
                "",
                "| Backend | CPU avg infer ms | CUDA avg infer ms | CUDA speedup vs CPU |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in summary["backend_ratios"]:
            lines.append(
                f"| {row['backend']} | {row['cpu_avg_infer_ms']:.2f} | {row['cuda_avg_infer_ms']:.2f} | {row['cuda_speedup_vs_cpu']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "backend",
        "device",
        "status",
        "selection_reason",
        "avg_infer_ms",
        "queries_per_sec",
        "init_s",
        "runtime_device",
        "runtime_dtype",
        "returncode",
        "reason",
        "case_output_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report-ready summary tables from a matrix artifact.")
    parser.add_argument("--matrix-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    matrix_json = args.matrix_json.expanduser().resolve()
    matrix = _load_matrix(matrix_json)
    matrix["matrix_source"] = str(matrix_json)

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else matrix_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _compute_summary(matrix)
    stamp = utc_stamp()
    out_json = output_dir / f"report_summary_{stamp}.json"
    out_md = output_dir / f"report_summary_{stamp}.md"
    out_csv = output_dir / f"report_summary_{stamp}.csv"

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_md.write_text(_build_markdown(summary), encoding="utf-8")
    _write_csv(out_csv, summary["cases"])

    print(json.dumps(summary, indent=2))
    print(f"\nWrote:\n- {out_json}\n- {out_md}\n- {out_csv}")


if __name__ == "__main__":
    main()
