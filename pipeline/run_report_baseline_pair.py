#!/usr/bin/env python3
"""
Run or plan the locked May-report baseline across one daytime and one nighttime clip.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_REVIEW_SCRIPT = REPO_ROOT / "pipeline" / "run_deployment_review.py"
DEFAULT_CONFIG = REPO_ROOT / "src" / "configuration-layer" / "config.report-baseline.yaml"
DEFAULT_REVIEW_ROOT = REPO_ROOT / "review-package"
DEFAULT_ARTIFACTS_DIR = DEFAULT_REVIEW_ROOT / "artifacts"
DEFAULT_DAY_VIDEO = REPO_ROOT / "data" / "upson1.mp4"
DEFAULT_NIGHT_VIDEO = REPO_ROOT / "data" / "sample1.mp4"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _rel_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _build_one_command(
    *,
    config_path: Path,
    video_path: Path,
    max_frames: int,
    config_tag: str,
    review_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(RUN_REVIEW_SCRIPT),
        "--config",
        _rel_to_repo(config_path),
        "--video",
        _rel_to_repo(video_path),
        "--config-tag",
        config_tag,
        "--review-root",
        _rel_to_repo(review_root),
    ]
    if max_frames > 0:
        command.extend(["--max-frames", str(max_frames)])
    return command


def _run_one(command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    duration_s = time.perf_counter() - started
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "duration_s": duration_s,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "status": "ok" if completed.returncode == 0 else "error",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or plan the locked May-report day/night baseline pair.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--day-video", type=Path, default=DEFAULT_DAY_VIDEO)
    parser.add_argument("--night-video", type=Path, default=DEFAULT_NIGHT_VIDEO)
    parser.add_argument("--day-tag", default="may_baseline_day")
    parser.add_argument("--night-tag", default="may_baseline_night")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    day_video = args.day_video.expanduser().resolve()
    night_video = args.night_video.expanduser().resolve()
    review_root = args.review_root.expanduser().resolve()
    artifacts_dir = args.artifacts_dir.expanduser().resolve()

    for path in (config_path, day_video, night_video, RUN_REVIEW_SCRIPT):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    review_root.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    planned_runs = [
        {
            "label": "day",
            "video": _rel_to_repo(day_video),
            "config_tag": str(args.day_tag),
            "selection_reason": "Chosen as the baseline daytime clip; brightest local recorded deployment-style clip available in the repo.",
            "command": _build_one_command(
                config_path=config_path,
                video_path=day_video,
                max_frames=int(args.max_frames),
                config_tag=str(args.day_tag),
                review_root=review_root,
            ),
        },
        {
            "label": "night",
            "video": _rel_to_repo(night_video),
            "config_tag": str(args.night_tag),
            "selection_reason": "Chosen as the baseline nighttime clip candidate; currently the darkest bundled local clip available in the repo.",
            "command": _build_one_command(
                config_path=config_path,
                video_path=night_video,
                max_frames=int(args.max_frames),
                config_tag=str(args.night_tag),
                review_root=review_root,
            ),
        },
    ]

    execution: list[dict[str, Any]] = []
    if not args.dry_run:
        for planned in planned_runs:
            result = _run_one(planned["command"])
            result["label"] = planned["label"]
            result["video"] = planned["video"]
            result["config_tag"] = planned["config_tag"]
            execution.append(result)
            if result["status"] != "ok":
                break

    payload = {
        "schema": "report_baseline_pair_v1",
        "created_at_utc": utc_iso(),
        "config": _rel_to_repo(config_path),
        "review_root": str(review_root),
        "artifacts_dir": str(artifacts_dir),
        "dry_run": bool(args.dry_run),
        "planned_runs": planned_runs,
        "execution": execution,
        "inference_note": "Day/night clip selection is an implementation inference from the currently available local repo videos and should be revised if a truer nighttime deployment clip is added.",
    }

    stamp = utc_stamp()
    out_json = artifacts_dir / f"report_baseline_pair_{stamp}.json"
    out_md = artifacts_dir / f"report_baseline_pair_{stamp}.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Report Baseline Pair",
        "",
        f"- Generated at: `{payload['created_at_utc']}`",
        f"- Config: `{payload['config']}`",
        f"- Dry run: `{payload['dry_run']}`",
        "",
        "## Planned Runs",
        "",
    ]
    for planned in planned_runs:
        lines.extend(
            [
                f"- `{planned['label']}`: `{planned['video']}`",
                f"  tag: `{planned['config_tag']}`",
                f"  reason: `{planned['selection_reason']}`",
            ]
        )
    if execution:
        lines.extend(["", "## Execution", ""])
        for result in execution:
            lines.extend(
                [
                    f"- `{result['label']}` status: `{result['status']}`",
                    f"  returncode: `{result['returncode']}`",
                    f"  duration_s: `{result['duration_s']:.2f}`",
                ]
            )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nWrote:\n- {out_json}\n- {out_md}")


if __name__ == "__main__":
    main()
