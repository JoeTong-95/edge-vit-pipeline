#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
VLM_DIR = THIS_DIR.parent
LAYER_PATH = VLM_DIR / "layer.py"
DEFAULT_IMAGE_PATH = VLM_DIR / "truckimage.png"
DEFAULT_SMOL_PATH = VLM_DIR / "SmolVLM-256M-Instruct"
DEFAULT_QWEN_PATH = VLM_DIR / "Qwen3.5-0.8B"
DEFAULT_GRACE_PATH = VLM_DIR / "grace_integration"


def _load_layer_module() -> Any:
    module_name = "vlm_layer_benchmark_runtime"
    spec = importlib.util.spec_from_file_location(module_name, LAYER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load layer module from {LAYER_PATH}")
    if str(VLM_DIR) not in sys.path:
        sys.path.insert(0, str(VLM_DIR))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _looks_like_git_lfs_pointer(file_path: Path) -> bool:
    if not file_path.is_file():
        return False
    try:
        head = file_path.read_text(encoding="utf-8", errors="ignore")[:200]
    except OSError:
        return False
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def _checkpoint_ready_for_hf(model_path: Path) -> tuple[bool, str]:
    if not model_path.exists():
        return False, f"missing model path: {model_path}"

    pointer_candidates = [
        model_path / "tokenizer.json",
        model_path / "model.safetensors",
        model_path / "model.safetensors-00001-of-00001.safetensors",
    ]
    flagged = [path.name for path in pointer_candidates if _looks_like_git_lfs_pointer(path)]
    if flagged:
        return False, "git-lfs pointer files present: " + ", ".join(flagged)
    return True, "ready"


def _checkpoint_ready_for_gemma(model_path: Path) -> tuple[bool, str]:
    if not model_path.exists():
        return False, f"missing model path: {model_path}"
    if model_path.is_file():
        if model_path.suffix.lower() != ".gguf":
            return False, f"expected .gguf file, got: {model_path.name}"
        mmproj_guess = model_path.parent / f"mmproj-{model_path.name}"
        if not mmproj_guess.exists():
            return False, f"missing mmproj file next to model: {mmproj_guess.name}"
        return True, "ready"

    model_files = sorted(path for path in model_path.glob("*.gguf") if "mmproj" not in path.name.lower())
    mmproj_files = sorted(path for path in model_path.glob("*.gguf") if "mmproj" in path.name.lower())
    if not model_files or not mmproj_files:
        return False, "expected model.gguf + mmproj-*.gguf in directory"
    return True, "ready"


def _checkpoint_ready_for_grace(model_path: Path) -> tuple[bool, str]:
    if not model_path.exists():
        return False, f"missing model path: {model_path}"
    required = [
        model_path / "inference.py",
        model_path / "config.yaml",
        model_path / "target_vehicle_types.yaml",
        model_path / "checkpoint" / "best_axle_graph_v6.pt",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        return False, "missing GRACE files: " + ", ".join(missing)
    return True, "ready"


def _parse_devices(raw: str) -> list[str]:
    devices = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not devices:
        raise ValueError("At least one device must be provided.")
    allowed = {"cpu", "cuda"}
    invalid = [item for item in devices if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported devices: {invalid}. Allowed: {sorted(allowed)}")
    return devices


def _parse_backends(raw: str) -> list[str]:
    backends = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not backends:
        raise ValueError("At least one backend must be provided.")
    allowed = {"smolvlm_256m", "qwen_0_8b", "gemma_e2b_local", "grace_fhwa"}
    invalid = [item for item in backends if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported backends: {invalid}. Allowed: {sorted(allowed)}")
    return backends


def _cleanup_runtime(layer: Any, runtime_state: Any) -> None:
    backend_kind = str(getattr(runtime_state, "vlm_runtime_backend_kind", ""))
    try:
        if backend_kind == "gemma_e2b_local":
            from backends import gemma_e2b_local

            gemma_e2b_local._shutdown_servers()
    except Exception:
        pass

    torch_mod = getattr(runtime_state, "vlm_runtime_torch", None)
    del runtime_state
    gc.collect()
    try:
        if torch_mod is not None and hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()
    except Exception:
        pass


def _run_one_case(
    *,
    layer: Any,
    backend_name: str,
    model_path: Path,
    device: str,
    image_path: Path,
    query_type: str,
    warmup_runs: int,
    measured_runs: int,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    crop_pkg = layer.VLMFrameCropperLayerPackage(
        vlm_frame_cropper_layer_track_id="benchmark-track-001",
        vlm_frame_cropper_layer_image=image,
        vlm_frame_cropper_layer_bbox=None,
    )

    t0 = time.perf_counter()
    runtime_state = layer.initialize_vlm_layer(
        layer.VLMConfig(
            config_vlm_enabled=True,
            config_vlm_backend=backend_name,
            config_vlm_model=str(model_path),
            config_device=device,
        )
    )
    init_s = time.perf_counter() - t0

    try:
        for _ in range(max(0, int(warmup_runs))):
            _ = layer.run_vlm_inference(
                vlm_runtime_state=runtime_state,
                vlm_frame_cropper_layer_package=crop_pkg,
                vlm_layer_query_type=query_type,
            )

        latencies_s: list[float] = []
        sample_text = ""
        for _ in range(max(1, int(measured_runs))):
            q0 = time.perf_counter()
            raw_result = layer.run_vlm_inference(
                vlm_runtime_state=runtime_state,
                vlm_frame_cropper_layer_package=crop_pkg,
                vlm_layer_query_type=query_type,
            )
            q1 = time.perf_counter()
            latencies_s.append(q1 - q0)
            if not sample_text:
                sample_text = str(raw_result.vlm_layer_raw_text).strip()

        total_infer_s = sum(latencies_s)
        avg_infer_s = total_infer_s / float(len(latencies_s))
        return {
            "status": "ok",
            "backend": backend_name,
            "device": device,
            "model_path": str(model_path),
            "resolved_model_id": str(runtime_state.vlm_runtime_model_id),
            "runtime_backend_kind": str(runtime_state.vlm_runtime_backend_kind),
            "runtime_device": str(runtime_state.vlm_runtime_device),
            "runtime_dtype": str(runtime_state.vlm_runtime_dtype),
            "init_s": init_s,
            "warmup_runs": int(warmup_runs),
            "measured_runs": int(len(latencies_s)),
            "avg_infer_ms": avg_infer_s * 1000.0,
            "min_infer_ms": min(latencies_s) * 1000.0,
            "max_infer_ms": max(latencies_s) * 1000.0,
            "queries_per_sec": (1.0 / avg_infer_s) if avg_infer_s > 0 else 0.0,
            "sample_text": sample_text[:240],
        }
    finally:
        _cleanup_runtime(layer, runtime_state)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark modular VLM backends across cpu/cuda with clear skip reasons."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument(
        "--backends",
        default="smolvlm_256m,qwen_0_8b,gemma_e2b_local,grace_fhwa",
        help="Comma-separated list from: smolvlm_256m,qwen_0_8b,gemma_e2b_local,grace_fhwa",
    )
    parser.add_argument("--devices", default="cpu,cuda", help="Comma-separated list from: cpu,cuda")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measured-runs", type=int, default=1)
    parser.add_argument("--query-type", default="vehicle_semantics_single_shot_v1")
    parser.add_argument("--smol-model", type=Path, default=DEFAULT_SMOL_PATH)
    parser.add_argument("--qwen-model", type=Path, default=DEFAULT_QWEN_PATH)
    parser.add_argument("--grace-model", type=Path, default=DEFAULT_GRACE_PATH)
    parser.add_argument(
        "--gemma-model",
        type=Path,
        default=Path(os.environ.get("GEMMA_E2B_LOCAL_MODEL", "")).expanduser() if os.environ.get("GEMMA_E2B_LOCAL_MODEL") else None,
        help="Path to local Gemma GGUF model file or directory (or set GEMMA_E2B_LOCAL_MODEL).",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    devices = _parse_devices(args.devices)
    requested_backends = set(_parse_backends(args.backends))
    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Benchmark image not found: {image_path}")

    layer = _load_layer_module()

    cases: list[tuple[str, Path | None, str]] = [
        ("smolvlm_256m", Path(args.smol_model).expanduser().resolve() if args.smol_model else None, "huggingface_local"),
        ("qwen_0_8b", Path(args.qwen_model).expanduser().resolve() if args.qwen_model else None, "huggingface_local"),
        ("gemma_e2b_local", Path(args.gemma_model).expanduser().resolve() if args.gemma_model else None, "gemma_e2b_local"),
        ("grace_fhwa", Path(args.grace_model).expanduser().resolve() if args.grace_model else None, "grace_fhwa"),
    ]

    results: list[dict[str, Any]] = []
    for backend_name, model_path, family in cases:
        if backend_name not in requested_backends:
            continue
        for device in devices:
            print(f"[matrix] start backend={backend_name} device={device}", file=sys.stderr, flush=True)
            if model_path is None:
                results.append(
                    {
                        "status": "skipped",
                        "backend": backend_name,
                        "device": device,
                        "reason": "no model path configured",
                    }
                )
                print(f"[matrix] skipped backend={backend_name} device={device}: no model path configured", file=sys.stderr, flush=True)
                continue

            if family == "huggingface_local":
                ready, reason = _checkpoint_ready_for_hf(model_path)
            elif family == "gemma_e2b_local":
                ready, reason = _checkpoint_ready_for_gemma(model_path)
            else:
                ready, reason = _checkpoint_ready_for_grace(model_path)

            if not ready:
                results.append(
                    {
                        "status": "skipped",
                        "backend": backend_name,
                        "device": device,
                        "model_path": str(model_path),
                        "reason": reason,
                    }
                )
                print(f"[matrix] skipped backend={backend_name} device={device}: {reason}", file=sys.stderr, flush=True)
                continue

            try:
                result = _run_one_case(
                    layer=layer,
                    backend_name=backend_name,
                    model_path=model_path,
                    device=device,
                    image_path=image_path,
                    query_type=args.query_type,
                    warmup_runs=args.warmup_runs,
                    measured_runs=args.measured_runs,
                )
            except Exception as exc:
                result = {
                    "status": "error",
                    "backend": backend_name,
                    "device": device,
                    "model_path": str(model_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            status = str(result.get("status", "unknown"))
            detail = str(result.get("error") or result.get("reason") or result.get("avg_infer_ms") or "")
            print(f"[matrix] done backend={backend_name} device={device} status={status} detail={detail}", file=sys.stderr, flush=True)
            results.append(result)

    payload = {
        "generated_at_unix_s": time.time(),
        "image_path": str(image_path),
        "devices": devices,
        "warmup_runs": int(args.warmup_runs),
        "measured_runs": int(args.measured_runs),
        "results": results,
    }

    print(json.dumps(payload, indent=2))

    if args.output_json is not None:
        out_path = args.output_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
