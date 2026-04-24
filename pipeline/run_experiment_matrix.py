#!/usr/bin/env python3
"""
Run the modular VLM backend/device matrix and save per-case + summary artifacts.

By default this runner skips combinations that are currently known to be risky on
this Jetson because prior attempts have been associated with SSH disconnects and
temporary host unreachability. Use --allow-unstable only for deliberate manual
follow-up work.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK_SCRIPT = REPO_ROOT / "src" / "vlm-layer" / "test" / "benchmark_vlm_backend_device_matrix.py"
DEFAULT_REVIEW_ROOT = REPO_ROOT / "review-package"
DEFAULT_ARTIFACTS_DIR = DEFAULT_REVIEW_ROOT / "artifacts"
DEFAULT_GEMMA_MODEL = REPO_ROOT / "src" / "vlm-layer" / "models" / "gemma-4-e2b-gguf"

KNOWN_UNSTABLE_CASES: dict[tuple[str, str], str] = {
    (
        "qwen_0_8b",
        "cpu",
    ): "Known unstable on this Jetson: prior attempts have hard-killed the process and were reported to coincide with SSH disconnect / temporary host loss.",
    (
        "qwen_0_8b",
        "cuda",
    ): "Known unstable on this Jetson: prior attempts have hard-killed the process and were reported to coincide with SSH disconnect / temporary host loss.",
    (
        "gemma_e2b_local",
        "cuda",
    ): "Known unstable / non-viable on this Jetson right now: prior attempts hit startup GPU-memory pressure and can destabilize the host session.",
}

CASE_PRESETS: dict[str, dict[str, Any]] = {
    "may_report_realistic": {
        "description": "Current report-stage recommendation on Jetson: compare the most realistic currently actionable VLM options first.",
        "cases": [
            {
                "backend": "smolvlm_256m",
                "device": "cpu",
                "selection_reason": "Stable measured reference path for the active modular Hugging Face backend.",
            },
            {
                "backend": "smolvlm_256m",
                "device": "cuda",
                "selection_reason": "Stable measured accelerated path for the active modular Hugging Face backend.",
            },
            {
                "backend": "gemma_e2b_local",
                "device": "cpu",
                "selection_reason": "Still a realistic local candidate to probe on CPU before attempting broader model/runtime changes.",
            },
        ],
    },
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_csv_values(raw: str, allowed: set[str], label: str) -> list[str]:
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"At least one {label} value is required.")
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"Unsupported {label}: {invalid}. Allowed: {sorted(allowed)}")
    return values


def _tail_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _markdown_lines(payload: dict[str, Any]) -> list[str]:
    lines = [
        "# VLM Backend Device Matrix Summary",
        "",
        f"- Generated at: `{payload['created_at_utc']}`",
        f"- Safe mode: `{payload['safe_mode']}`",
        f"- Allow unstable: `{payload['allow_unstable']}`",
        f"- Benchmark script: `{payload['benchmark_script']}`",
        f"- Output root: `{payload['artifacts_dir']}`",
        "",
        "## Cases",
        "",
    ]
    for case in payload["cases"]:
        label = f"{case['backend']} on {case['device']}"
        status = case.get("status", "unknown")
        lines.append(f"- `{label}`: `{status}`")
        detail = (
            case.get("reason")
            or case.get("error")
            or case.get("skip_reason")
            or case.get("avg_infer_ms")
            or case.get("returncode")
        )
        if detail not in (None, ""):
            lines.append(f"  detail: `{detail}`")
    lines.append("")
    return lines


def _build_case_command(
    *,
    benchmark_script: Path,
    backend: str,
    device: str,
    image: Path,
    warmup_runs: int,
    measured_runs: int,
    query_type: str,
    case_output_json: Path,
    smol_model: Path | None,
    qwen_model: Path | None,
    gemma_model: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(benchmark_script),
        "--backends",
        backend,
        "--devices",
        device,
        "--image",
        str(image),
        "--warmup-runs",
        str(warmup_runs),
        "--measured-runs",
        str(measured_runs),
        "--query-type",
        query_type,
        "--output-json",
        str(case_output_json),
    ]
    if smol_model is not None:
        command.extend(["--smol-model", str(smol_model)])
    if qwen_model is not None:
        command.extend(["--qwen-model", str(qwen_model)])
    if gemma_model is not None:
        command.extend(["--gemma-model", str(gemma_model)])
    return command


def _resolve_requested_cases(
    *,
    preset: str,
    backends: list[str],
    devices: list[str],
) -> tuple[list[dict[str, str]], str]:
    if preset:
        preset_payload = CASE_PRESETS[preset]
        return list(preset_payload["cases"]), str(preset_payload["description"])

    cases: list[dict[str, str]] = []
    for backend in backends:
        for device in devices:
            cases.append({"backend": backend, "device": device, "selection_reason": "manual_cli_selection"})
    return cases, "manual backend/device selection"


def _run_case(
    *,
    benchmark_script: Path,
    backend: str,
    device: str,
    image: Path,
    warmup_runs: int,
    measured_runs: int,
    query_type: str,
    timeout_s: int,
    artifacts_dir: Path,
    smol_model: Path | None,
    qwen_model: Path | None,
    gemma_model: Path | None,
) -> dict[str, Any]:
    case_output_json = artifacts_dir / f"{backend}__{device}.json"
    command = _build_case_command(
        benchmark_script=benchmark_script,
        backend=backend,
        device=device,
        image=image,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        query_type=query_type,
        case_output_json=case_output_json,
        smol_model=smol_model,
        qwen_model=qwen_model,
        gemma_model=gemma_model,
    )

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=dict(os.environ),
    )
    duration_s = time.perf_counter() - started

    payload = _load_json_if_present(case_output_json)
    result_row = None
    if payload:
        rows = payload.get("results") or []
        if rows:
            result_row = rows[0]

    result: dict[str, Any] = {
        "backend": backend,
        "device": device,
        "timeout_s": timeout_s,
        "duration_s": duration_s,
        "returncode": int(completed.returncode),
        "stdout_tail": _tail_text(completed.stdout),
        "stderr_tail": _tail_text(completed.stderr),
        "json_exists": case_output_json.exists(),
        "case_output_json": str(case_output_json),
    }
    if payload is not None:
        result["payload"] = payload
    if isinstance(result_row, dict):
        result.update(result_row)
    elif completed.returncode == 0:
        result["status"] = "ok"
    else:
        result["status"] = "error"
        result["error"] = f"benchmark subprocess exited with return code {completed.returncode}"

    # Promote likely host-instability symptoms into the recorded reason.
    if int(completed.returncode) == -9 and not case_output_json.exists():
        result.setdefault(
            "reason",
            "Process was hard-killed before result JSON was written; on this Jetson this case has also been associated with SSH disconnect / temporary host loss.",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the report-stage VLM backend/device matrix safely.")
    parser.add_argument(
        "--backends",
        default="smolvlm_256m,qwen_0_8b,gemma_e2b_local",
        help="Comma-separated list from: smolvlm_256m,qwen_0_8b,gemma_e2b_local",
    )
    parser.add_argument("--devices", default="cpu,cuda", help="Comma-separated list from: cpu,cuda")
    parser.add_argument("--image", type=Path, default=REPO_ROOT / "src" / "vlm-layer" / "truckimage.png")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measured-runs", type=int, default=1)
    parser.add_argument("--query-type", default="vehicle_semantics_single_shot_v1")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--benchmark-script", type=Path, default=DEFAULT_BENCHMARK_SCRIPT)
    parser.add_argument("--smol-model", type=Path, default=REPO_ROOT / "src" / "vlm-layer" / "SmolVLM-256M-Instruct")
    parser.add_argument("--qwen-model", type=Path, default=REPO_ROOT / "src" / "vlm-layer" / "Qwen3.5-0.8B")
    parser.add_argument("--gemma-model", type=Path, default=DEFAULT_GEMMA_MODEL if DEFAULT_GEMMA_MODEL.exists() else None)
    parser.add_argument(
        "--allow-unstable",
        action="store_true",
        help="Include combinations that are currently known to destabilize this Jetson.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(CASE_PRESETS.keys()),
        default="",
        help="Optional named case selection preset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve case selection and write summary artifacts without executing benchmarks.",
    )
    args = parser.parse_args()

    backends = _parse_csv_values(
        args.backends,
        {"smolvlm_256m", "qwen_0_8b", "gemma_e2b_local"},
        "backends",
    )
    devices = _parse_csv_values(args.devices, {"cpu", "cuda"}, "devices")

    benchmark_script = args.benchmark_script.expanduser().resolve()
    if not benchmark_script.exists():
        raise FileNotFoundError(f"Benchmark script not found: {benchmark_script}")

    image = args.image.expanduser().resolve()
    if not image.exists():
        raise FileNotFoundError(f"Benchmark image not found: {image}")

    review_root = args.review_root.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    review_root.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    smol_model = args.smol_model.expanduser().resolve() if args.smol_model else None
    qwen_model = args.qwen_model.expanduser().resolve() if args.qwen_model else None
    gemma_model = args.gemma_model.expanduser().resolve() if args.gemma_model else None

    requested_cases, preset_description = _resolve_requested_cases(
        preset=str(args.preset or "").strip(),
        backends=backends,
        devices=devices,
    )

    cases: list[dict[str, Any]] = []
    for requested in requested_cases:
        backend = str(requested["backend"])
        device = str(requested["device"])
        selection_reason = str(requested.get("selection_reason", "")).strip()
        case_record_base = {
            "backend": backend,
            "device": device,
            "selection_reason": selection_reason,
        }

        if args.dry_run:
            unstable_reason = KNOWN_UNSTABLE_CASES.get((backend, device))
            cases.append(
                {
                    **case_record_base,
                    "status": "planned",
                    "safe_to_run_now": not bool(unstable_reason),
                    "skip_reason": unstable_reason or "",
                }
            )
            continue

        unstable_reason = KNOWN_UNSTABLE_CASES.get((backend, device))
        if unstable_reason and not args.allow_unstable:
            unstable_reason = KNOWN_UNSTABLE_CASES.get((backend, device))
            cases.append(
                {
                    **case_record_base,
                    "status": "skipped",
                    "skip_reason": unstable_reason,
                }
            )
            continue
        try:
            result = _run_case(
                benchmark_script=benchmark_script,
                backend=backend,
                device=device,
                image=image,
                warmup_runs=int(args.warmup_runs),
                measured_runs=int(args.measured_runs),
                query_type=str(args.query_type),
                timeout_s=int(args.timeout_s),
                artifacts_dir=artifacts_dir,
                smol_model=smol_model,
                qwen_model=qwen_model,
                gemma_model=gemma_model,
            )
        except subprocess.TimeoutExpired as exc:
            result = {
                "backend": backend,
                "device": device,
                "status": "error",
                "timeout_s": int(args.timeout_s),
                "error": f"benchmark timed out after {int(args.timeout_s)} seconds",
                "stdout_tail": _tail_text(exc.stdout or ""),
                "stderr_tail": _tail_text(exc.stderr or ""),
            }
        result["selection_reason"] = selection_reason
        cases.append(result)

    payload = {
        "schema": "vlm_backend_device_matrix_runner_v1",
        "created_at_utc": utc_iso(),
        "generated_at_unix_s": time.time(),
        "review_root": str(review_root),
        "artifacts_dir": str(artifacts_dir),
        "benchmark_script": str(benchmark_script),
        "image_path": str(image),
        "warmup_runs": int(args.warmup_runs),
        "measured_runs": int(args.measured_runs),
        "timeout_s": int(args.timeout_s),
        "backends": backends,
        "devices": devices,
        "preset": str(args.preset or ""),
        "preset_description": preset_description,
        "dry_run": bool(args.dry_run),
        "allow_unstable": bool(args.allow_unstable),
        "safe_mode": not bool(args.allow_unstable),
        "cases": cases,
    }

    stamp = utc_stamp()
    summary_json = artifacts_dir / f"vlm_backend_device_matrix_summary_{stamp}.json"
    summary_md = artifacts_dir / f"vlm_backend_device_matrix_summary_{stamp}.md"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_md.write_text("\n".join(_markdown_lines(payload)), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote:\n- {summary_json}\n- {summary_md}")


if __name__ == "__main__":
    main()
