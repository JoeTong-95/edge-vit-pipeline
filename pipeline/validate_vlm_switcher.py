#!/usr/bin/env python3
"""
Safely validate config-driven VLM switcher resolution without running inference.

This helper is meant to be used before risky Jetson smoke runs. It reports:
- whether config validation passes
- which VLM backend name resolves from the config
- which runtime backend kind resolves from the config
- which device/runtime mode the pipeline would attempt to use
- whether the referenced checkpoint path looks locally runnable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
CONFIG_DIR = SRC_DIR / "configuration-layer"
VLM_DIR = SRC_DIR / "vlm-layer"
VLM_BACKENDS_DIR = VLM_DIR / "backends"

for path in (CONFIG_DIR, VLM_DIR, VLM_BACKENDS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _resolve_repo_path(path_value: str) -> Path:
    raw = Path(str(path_value or "").strip())
    if raw.is_absolute():
        return raw.resolve()
    repo_candidate = (REPO_ROOT / raw).resolve()
    if repo_candidate.exists():
        return repo_candidate
    vlm_candidate = (VLM_DIR / raw).resolve()
    if vlm_candidate.exists():
        return vlm_candidate
    return repo_candidate


def _looks_like_git_lfs_pointer(file_path: Path) -> bool:
    if not file_path.is_file():
        return False
    try:
        with file_path.open("rb") as fh:
            head_bytes = fh.read(200)
    except OSError:
        return False
    head = head_bytes.decode("utf-8", errors="ignore")
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def _checkpoint_readiness(runtime_kind: str, model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        return {"ready": False, "reason": f"missing model path: {model_path}"}

    if runtime_kind == "huggingface_local":
        config_path = model_path / "config.json"
        if not config_path.exists():
            return {"ready": False, "reason": "missing config.json in Hugging Face model directory"}
        try:
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"ready": False, "reason": f"unable to parse config.json: {exc}"}
        if "model_type" not in config_payload:
            return {"ready": False, "reason": "config.json is present but missing model_type"}
        pointer_candidates = [
            model_path / "tokenizer.json",
            model_path / "model.safetensors",
            model_path / "model.safetensors-00001-of-00001.safetensors",
        ]
        flagged = [path.name for path in pointer_candidates if _looks_like_git_lfs_pointer(path)]
        if flagged:
            return {"ready": False, "reason": "git-lfs pointer files present: " + ", ".join(flagged)}
        return {"ready": True, "reason": "checkpoint path present and no obvious git-lfs pointers detected"}

    if runtime_kind == "gemma_e2b_local":
        if model_path.is_file():
            if model_path.suffix.lower() != ".gguf":
                return {"ready": False, "reason": f"expected .gguf file, got: {model_path.name}"}
            mmproj_guess = model_path.parent / f"mmproj-{model_path.name}"
            if not mmproj_guess.exists():
                return {"ready": False, "reason": f"missing mmproj file: {mmproj_guess.name}"}
            return {"ready": True, "reason": "gguf model + mmproj file present"}

        model_files = sorted(path for path in model_path.glob("*.gguf") if "mmproj" not in path.name.lower())
        mmproj_files = sorted(path for path in model_path.glob("*.gguf") if "mmproj" in path.name.lower())
        if not model_files or not mmproj_files:
            return {"ready": False, "reason": "expected model.gguf + mmproj*.gguf in directory"}
        return {"ready": True, "reason": "gguf directory contains model + mmproj files"}

    return {"ready": False, "reason": f"unsupported runtime kind for readiness check: {runtime_kind}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely validate config-driven VLM switcher resolution.")
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "config.yaml"),
        help="Repo-relative or absolute path to config YAML.",
    )
    args = parser.parse_args()

    from config_node import get_config_value, load_config, validate_config
    from registry import resolve_vlm_backend_name, resolve_vlm_backend_runtime_kind

    config_path = _resolve_repo_path(str(args.config))
    if not config_path.exists():
        raise FileNotFoundError(f"Config path not found: {config_path}")

    config = load_config(config_path)
    validate_config(config)

    config_device = str(get_config_value(config, "config_device")).strip().lower()
    vlm_enabled = bool(get_config_value(config, "config_vlm_enabled"))
    config_vlm_backend = str(get_config_value(config, "config_vlm_backend") or "auto").strip().lower()
    config_vlm_model = str(get_config_value(config, "config_vlm_model") or "").strip()
    config_vlm_device = str(get_config_value(config, "config_vlm_device") or "").strip().lower()
    config_vlm_runtime_mode = str(get_config_value(config, "config_vlm_runtime_mode") or "inline").strip().lower()

    effective_device = config_vlm_device if config_vlm_device else config_device
    resolved_model_path = _resolve_repo_path(config_vlm_model)
    resolved_backend_name = resolve_vlm_backend_name(config_vlm_backend, config_vlm_model) if vlm_enabled else "disabled"
    resolved_runtime_kind = (
        resolve_vlm_backend_runtime_kind(config_vlm_backend, config_vlm_model) if vlm_enabled else "disabled"
    )
    readiness = (
        _checkpoint_readiness(resolved_runtime_kind, resolved_model_path)
        if vlm_enabled
        else {"ready": False, "reason": "vlm disabled"}
    )

    payload = {
        "schema": "vlm_switcher_validation_v1",
        "config_path": str(config_path),
        "vlm_enabled": vlm_enabled,
        "config_device": config_device,
        "config_vlm_backend": config_vlm_backend,
        "config_vlm_model": config_vlm_model,
        "resolved_model_path": str(resolved_model_path),
        "config_vlm_device": config_vlm_device,
        "effective_vlm_device": effective_device,
        "config_vlm_runtime_mode": config_vlm_runtime_mode,
        "resolved_backend_name": resolved_backend_name,
        "resolved_runtime_kind": resolved_runtime_kind,
        "checkpoint_readiness": readiness,
        "jetson_risk_note": (
            "This helper does not run inference. Previous unstable backend/device smoke runs on this Jetson "
            "were associated with SSH disconnect / temporary host loss."
        ),
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
