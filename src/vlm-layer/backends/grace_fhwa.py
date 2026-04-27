from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image


VLM_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GRACE_DIR = VLM_DIR / "grace_integration"
DEFAULT_TARGET_TYPES_PATH = DEFAULT_GRACE_DIR / "target_vehicle_types.yaml"


def initialize_backend(
    *,
    model_path: str,
    requested_device: str,
) -> dict[str, Any]:
    package_dir, config_path, checkpoint_path = _resolve_grace_paths(model_path)
    inference_module = _load_grace_inference_module(package_dir)
    model, runtime_device = inference_module.load_model(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        device=_normalize_requested_device(requested_device),
    )

    return {
        "backend_name": "grace_fhwa",
        "model_path": str(package_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "target_vehicle_types_path": str(package_dir / "target_vehicle_types.yaml"),
        "target_vehicle_types": load_target_vehicle_types(package_dir / "target_vehicle_types.yaml"),
        "runtime_device": str(runtime_device),
        "runtime_dtype": "torch.float32",
        "model_id": f"GRACE:{checkpoint_path.name}",
        "model": model,
        "inference_module": inference_module,
    }


def infer_single(
    *,
    backend_state: dict[str, Any],
    image: Any,
    prompt_text: str,
) -> str:
    del prompt_text
    return infer_batch(
        backend_state=backend_state,
        images=[image],
        prompt_texts=[""],
    )[0]


def infer_batch(
    *,
    backend_state: dict[str, Any],
    images: list[Any],
    prompt_texts: list[str],
) -> list[str]:
    del prompt_texts
    if not images:
        return []

    module = backend_state["inference_module"]
    torch = module.torch
    transform = _build_preprocess_transform(module)
    tensors = [transform(_coerce_rgb_image(image)) for image in images]
    model = backend_state["model"]
    device = next(model.parameters()).device
    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        outputs = model(batch, return_heatmap=False)

    fhwa_indices = outputs["fhwa_logits"].argmax(dim=-1)
    primary_indices = outputs["primary_logits"].argmax(dim=-1)
    trailer_indices = outputs["trailer_logits"].argmax(dim=-1)
    axle_counts = module.predict_from_ordinal_logits(
        outputs["axle_count_logits"],
        method="expectation",
    )
    fhwa_probs = torch.softmax(outputs["fhwa_logits"], dim=-1)

    target_vehicle_types = set(backend_state.get("target_vehicle_types") or [])
    results: list[str] = []
    for index in range(len(images)):
        fhwa_index = int(fhwa_indices[index].item())
        primary_index = int(primary_indices[index].item())
        trailer_index = int(trailer_indices[index].item())
        grace_result = {
            "fhwa_class": module.FHWA_NAMES.get(fhwa_index, f"IDX-{fhwa_index}"),
            "fhwa_index": fhwa_index,
            "fhwa_confidence": round(float(fhwa_probs[index, fhwa_index].item()), 4),
            "vehicle_type": module.VEHICLE_TYPES.get(primary_index, f"IDX-{primary_index}"),
            "vehicle_type_index": primary_index,
            "trailer_count": module.TRAILER_NAMES.get(trailer_index, str(trailer_index)),
            "axle_count": round(float(axle_counts[index].item()), 2),
        }
        results.append(
            convert_grace_result_to_vlm_json(
                grace_result,
                target_vehicle_types=target_vehicle_types,
            )
        )
    return results


def convert_grace_result_to_vlm_json(
    grace_result: dict[str, Any],
    *,
    target_vehicle_types: set[str] | list[str] | tuple[str, ...] | None = None,
) -> str:
    targets = set(target_vehicle_types or load_target_vehicle_types(DEFAULT_TARGET_TYPES_PATH))
    vehicle_type = str(grace_result.get("vehicle_type") or "unknown").strip().lower()
    confidence = _coerce_float(grace_result.get("fhwa_confidence"), default=0.0)
    is_target_vehicle = vehicle_type in targets
    payload = {
        "is_target_vehicle": is_target_vehicle,
        "ack_status": "accepted",
        "retry_reasons": [],
        "confidence": round(confidence, 4),
        "grace_backend": "grace_fhwa",
        "fhwa_class": grace_result.get("fhwa_class"),
        "fhwa_index": grace_result.get("fhwa_index"),
        "fhwa_confidence": round(confidence, 4),
        "vehicle_type": grace_result.get("vehicle_type"),
        "vehicle_type_index": grace_result.get("vehicle_type_index"),
        "axle_count": _coerce_float(grace_result.get("axle_count"), default=0.0),
        "trailer_count": grace_result.get("trailer_count"),
    }
    return json.dumps(payload, separators=(",", ":"))


def load_target_vehicle_types(path: str | Path = DEFAULT_TARGET_TYPES_PATH) -> set[str]:
    target_path = Path(path)
    if not target_path.is_file():
        raise FileNotFoundError(f"GRACE target vehicle type YAML not found: {target_path}")
    try:
        import yaml
    except ModuleNotFoundError:
        return _load_target_vehicle_types_without_yaml(target_path)

    with open(target_path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    values = payload.get("target_vehicle_types") if isinstance(payload, dict) else None
    if not isinstance(values, list):
        raise ValueError("target_vehicle_types.yaml must contain a target_vehicle_types list.")
    return _normalize_target_vehicle_types(values)


def _load_target_vehicle_types_without_yaml(path: Path) -> set[str]:
    values: list[str] = []
    in_list = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.strip() == "target_vehicle_types:":
            in_list = True
            continue
        if in_list and line.lstrip().startswith("- "):
            values.append(line.lstrip()[2:].strip())
    return _normalize_target_vehicle_types(values)


def _normalize_target_vehicle_types(values: list[Any]) -> set[str]:
    normalized = {str(value).strip().lower() for value in values if str(value).strip()}
    if not normalized:
        raise ValueError("target_vehicle_types.yaml must define at least one target vehicle type.")
    return normalized


def _resolve_grace_paths(model_path: str) -> tuple[Path, Path, Path]:
    raw = str(model_path or "").strip()
    candidate = Path(raw).expanduser() if raw else DEFAULT_GRACE_DIR
    if not candidate.is_absolute():
        candidate = (VLM_DIR / candidate).resolve() if not (Path.cwd() / candidate).exists() else (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if candidate.is_file() and candidate.suffix == ".pt":
        checkpoint_path = candidate
        package_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoint" else checkpoint_path.parent
    else:
        package_dir = candidate
        checkpoint_path = package_dir / "checkpoint" / "best_axle_graph_v6.pt"

    config_path = package_dir / "config.yaml"
    inference_path = package_dir / "inference.py"
    for required_path in (package_dir, inference_path, config_path, checkpoint_path):
        if not required_path.exists():
            raise FileNotFoundError(f"GRACE backend path was not found: {required_path}")
    return package_dir, config_path, checkpoint_path


def _load_grace_inference_module(package_dir: Path) -> Any:
    inference_path = package_dir / "inference.py"
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    spec = importlib.util.spec_from_file_location("grace_fhwa_inference", inference_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load GRACE inference module from {inference_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_requested_device(requested_device: str) -> str:
    return "cuda" if requested_device == "auto" else str(requested_device or "cpu")


def _build_preprocess_transform(module: Any) -> Any:
    return module.transforms.Compose([
        module.transforms.Resize((224, 224)),
        module.transforms.ToTensor(),
        module.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _coerce_rgb_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
