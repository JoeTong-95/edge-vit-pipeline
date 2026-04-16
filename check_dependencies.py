#!/usr/bin/env python3
"""
check_dependencies.py (repo root)

Checks local prerequisites for running the edge VLM pipeline without executing
any benchmark or pipeline run.

What it checks:
  - Python version (basic sanity)
  - Key Python imports used by this repo
  - `src/configuration-layer/config.yaml` load + validate (via config layer)
  - Configured video path exists (video-only pipeline)
  - Optional: VLM model path exists (if VLM enabled)

It prints actionable install hints (pip commands) when something is missing.

Usage (from repo root):
  python check_dependencies.py
  python check_dependencies.py --requirements docker/requirements.dev.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
DEFAULT_REQUIREMENTS = REPO_ROOT / "docker" / "requirements.dev.txt"

CONFIG_DIR = SRC_DIR / "configuration-layer"

# Keep this list small + pragmatic: it should reflect what users typically need
# to run the pipeline locally.
_IMPORT_CHECKS: tuple[tuple[str, str], ...] = (
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("yaml", "PyYAML"),
    ("ultralytics", "ultralytics"),
    ("supervision", "supervision"),
    ("torch", "torch"),
    ("transformers", "transformers"),
)


def _resolve_repo_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _try_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def _missing_imports() -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for mod, pkg in _IMPORT_CHECKS:
        if not _try_import(mod):
            missing.append((mod, pkg))
    return missing


def _print_pip_hints(requirements: Path | None, missing: Iterable[tuple[str, str]]) -> None:
    missing_pkgs = sorted({pkg for _, pkg in missing})
    if requirements is not None and requirements.is_file():
        rel = requirements
        try:
            rel = requirements.relative_to(REPO_ROOT)
        except Exception:
            pass
        print()
        print("Install suggestion (recommended):")
        print(f'  {sys.executable} -m pip install -r \"{rel}\"')
    if missing_pkgs:
        print()
        print("Install suggestion (minimal):")
        print(f"  {sys.executable} -m pip install " + " ".join(missing_pkgs))


def _check_config_and_paths(video_override: str) -> tuple[bool, list[str]]:
    """
    Returns (ok, messages). Uses the repo's configuration layer to validate config.yaml.
    """
    messages: list[str] = []

    if str(CONFIG_DIR) not in sys.path:
        sys.path.insert(0, str(CONFIG_DIR))

    try:
        from config_node import get_config_value, load_config, validate_config
    except Exception as exc:
        messages.append(f"[config] FAILED to import configuration layer ({type(exc).__name__}: {exc})")
        return False, messages

    config_path = CONFIG_DIR / "config.yaml"
    if not config_path.is_file():
        messages.append(f"[config] MISSING {config_path}")
        return False, messages

    try:
        cfg = load_config(config_path)
        validate_config(cfg)
    except Exception as exc:
        messages.append(f"[config] INVALID ({type(exc).__name__}: {exc})")
        return False, messages

    try:
        input_source = str(get_config_value(cfg, "config_input_source"))
        if input_source != "video":
            messages.append(f"[config] NOTE config_input_source={input_source!r} (this repo is primarily video-first)")
    except Exception as exc:
        messages.append(f"[config] FAILED reading config_input_source ({type(exc).__name__}: {exc})")
        return False, messages

    try:
        configured_video_path = str(get_config_value(cfg, "config_input_path"))
        video_path_value = video_override.strip() or configured_video_path
        video_path = _resolve_repo_path(video_path_value)
        if not os.path.exists(video_path):
            messages.append(f"[data] MISSING configured video path: {video_path}")
            return False, messages
        messages.append(f"[data] OK video path: {video_path}")
    except Exception as exc:
        messages.append(f"[data] FAILED reading config_input_path ({type(exc).__name__}: {exc})")
        return False, messages

    try:
        vlm_enabled = bool(get_config_value(cfg, "config_vlm_enabled"))
        if vlm_enabled:
            vlm_model = str(get_config_value(cfg, "config_vlm_model")).strip()
            if vlm_model:
                vlm_model_path = _resolve_repo_path(vlm_model)
                if not os.path.exists(vlm_model_path):
                    messages.append(f"[vlm] MISSING model path: {vlm_model_path}")
                    return False, messages
                messages.append(f"[vlm] OK model path: {vlm_model_path}")
            else:
                messages.append("[vlm] ENABLED but config_vlm_model is empty")
                return False, messages
        else:
            messages.append("[vlm] disabled in config")
    except Exception as exc:
        messages.append(f"[vlm] FAILED reading VLM config keys ({type(exc).__name__}: {exc})")
        return False, messages

    return True, messages


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check pipeline prerequisites (imports + config + model/video paths). Does NOT run benchmark."
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=DEFAULT_REQUIREMENTS,
        help=f"Requirements file for install hints (default: {DEFAULT_REQUIREMENTS})",
    )
    parser.add_argument("--video", default="", help="Override config_input_path (repo-relative or absolute).")
    args = parser.parse_args()

    ok = True
    if sys.version_info < (3, 10):
        ok = False
        print(f"[python] FAIL version={sys.version.split()[0]} (need >= 3.10)")
    else:
        print(f"[python] OK version={sys.version.split()[0]}")

    req_path = args.requirements if args.requirements.is_absolute() else (REPO_ROOT / args.requirements)

    missing = _missing_imports()
    if missing:
        ok = False
        print()
        print("[imports] FAIL missing:")
        for mod, pkg in missing:
            print(f"  - import {mod!r}  (pip: {pkg})")
        _print_pip_hints(req_path, missing)
    else:
        print("[imports] OK")

    cfg_ok, cfg_msgs = _check_config_and_paths(video_override=args.video)
    if not cfg_ok:
        ok = False
        print()
        for m in cfg_msgs:
            print(m)
    else:
        print()
        for m in cfg_msgs:
            print(m)

    print()
    if ok:
        print("[check] OK (prerequisites look good).")
        return 0
    print("[check] FAILED (see messages above).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

