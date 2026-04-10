from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


LAYER_DIR = Path(__file__).resolve().parent
DEFAULT_VLM_MODEL_PATH = LAYER_DIR / "Qwen3.5-0.8B"
DEFAULT_VLM_QUERY_TYPE = "vehicle_semantics_v1"
_TRANSFORMERS_UPGRADE_HINT = (
    'pip install -U "transformers>=5.0.0,<6.0.0"'
)
_ALLOWED_VLM_ACK_STATUSES = {"accepted", "retry_requested", "finalize_with_current"}
_ALLOWED_VLM_RETRY_REASONS = {
    "occluded",
    "bad_angle",
}


@dataclass(slots=True)
class VLMConfig:
    config_vlm_enabled: bool = False
    config_vlm_model: str = str(DEFAULT_VLM_MODEL_PATH)
    config_device: str = "auto"


@dataclass(slots=True)
class VLMFrameCropperLayerPackage:
    vlm_frame_cropper_layer_track_id: str
    vlm_frame_cropper_layer_image: Image.Image | Any
    vlm_frame_cropper_layer_bbox: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class VLMRawResult:
    vlm_layer_track_id: str
    vlm_layer_query_type: str
    vlm_layer_model_id: str
    vlm_layer_raw_text: str
    vlm_layer_raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VLMLayerPackage:
    vlm_layer_track_id: str
    vlm_layer_query_type: str
    vlm_layer_label: str
    vlm_layer_attributes: dict[str, Any]
    vlm_layer_confidence: float | None
    vlm_layer_model_id: str


@dataclass(slots=True)
class VLMAckPackage:
    vlm_ack_track_id: str
    vlm_ack_status: str
    vlm_ack_reason: str
    vlm_ack_retry_requested: bool = False


@dataclass(slots=True)
class VLMRuntimeState:
    config_vlm_enabled: bool
    config_vlm_model: str
    config_device: str
    vlm_runtime_device: str
    vlm_runtime_dtype: str
    vlm_runtime_model_id: str
    vlm_runtime_processor: Any = None
    vlm_runtime_model: Any = None
    vlm_runtime_torch: Any = None


def initialize_vlm_layer(config: VLMConfig) -> VLMRuntimeState:
    """Load the configured VLM and return a runtime state package."""
    if not config.config_vlm_enabled:
        return VLMRuntimeState(
            config_vlm_enabled=False,
            config_vlm_model=config.config_vlm_model,
            config_device=config.config_device,
            vlm_runtime_device="disabled",
            vlm_runtime_dtype="disabled",
            vlm_runtime_model_id=Path(config.config_vlm_model).name or "disabled",
        )

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required to initialize the VLM layer at runtime."
        ) from exc

    try:
        import transformers
        from transformers import AutoProcessor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "transformers is required to initialize the VLM layer."
        ) from exc

    model_path = Path(config.config_vlm_model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Configured VLM model path was not found: {model_path}")

    _maybe_require_newer_transformers_for_checkpoint(model_path=model_path)

    runtime_device = _resolve_device(torch=torch, requested_device=config.config_device)
    runtime_dtype = torch.float16 if runtime_device == "cuda" else torch.float32

    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model_class = _load_model_class(transformers)
        model = model_class.from_pretrained(
            model_path,
            torch_dtype=runtime_dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
    except (KeyError, ValueError) as exc:
        _raise_if_unsupported_qwen35_model_type(exc)
        raise
    model = model.to(runtime_device)
    model.eval()

    return VLMRuntimeState(
        config_vlm_enabled=True,
        config_vlm_model=str(model_path),
        config_device=config.config_device,
        vlm_runtime_device=runtime_device,
        vlm_runtime_dtype=str(runtime_dtype),
        vlm_runtime_model_id=model_path.name,
        vlm_runtime_processor=processor,
        vlm_runtime_model=model,
        vlm_runtime_torch=torch,
    )


def run_vlm_inference(
    vlm_runtime_state: VLMRuntimeState,
    vlm_frame_cropper_layer_package: VLMFrameCropperLayerPackage,
    vlm_layer_query_type: str = DEFAULT_VLM_QUERY_TYPE,
) -> VLMRawResult:
    """Run semantic inference on one crop package and return the raw result."""
    if not vlm_runtime_state.config_vlm_enabled:
        raise RuntimeError("The VLM layer is disabled, so inference cannot run.")

    prompt_text = prepare_vlm_prompt(
        vlm_layer_query_type=vlm_layer_query_type,
        vlm_frame_cropper_layer_package=vlm_frame_cropper_layer_package,
    )
    raw_text = infer_vlm_semantics(
        vlm_runtime_state=vlm_runtime_state,
        vlm_frame_cropper_layer_package=vlm_frame_cropper_layer_package,
        vlm_prompt_text=prompt_text,
    )

    return VLMRawResult(
        vlm_layer_track_id=vlm_frame_cropper_layer_package.vlm_frame_cropper_layer_track_id,
        vlm_layer_query_type=vlm_layer_query_type,
        vlm_layer_model_id=vlm_runtime_state.vlm_runtime_model_id,
        vlm_layer_raw_text=raw_text,
        vlm_layer_raw_response={"prompt_text": prompt_text},
    )


def normalize_vlm_result(vlm_layer_raw_result: VLMRawResult) -> dict[str, Any]:
    """Convert the raw model output into stable normalized semantic fields."""
    parsed_fields = parse_vlm_response(vlm_layer_raw_result.vlm_layer_raw_text)
    parsed_fields.setdefault("is_truck", True)
    parsed_fields.setdefault("truck_type", "unknown")
    parsed_fields.setdefault("wheel_count", "unknown")
    parsed_fields.setdefault("estimated_weight_kg", "unknown")
    parsed_fields.setdefault("vlm_ack_status", "retry_requested")
    parsed_fields.setdefault("vlm_retry_reasons", ["occluded"])
    parsed_fields.setdefault("vlm_image_quality_notes", "")
    parsed_fields.setdefault("vlm_layer_confidence", None)

    parsed_fields["vlm_layer_label"] = "not_truck" if parsed_fields.get("is_truck") is False else parsed_fields.get("truck_type", "unknown")
    parsed_fields["vlm_layer_attributes"] = {
        "is_truck": parsed_fields.get("is_truck", True),
        "truck_type": parsed_fields.get("truck_type", "unknown"),
        "wheel_count": parsed_fields.get("wheel_count", "unknown"),
        "estimated_weight_kg": parsed_fields.get("estimated_weight_kg", "unknown"),
        "vlm_ack_status": parsed_fields.get("vlm_ack_status", "retry_requested"),
        "vlm_retry_reasons": list(parsed_fields.get("vlm_retry_reasons", [])),
        "vlm_image_quality_notes": parsed_fields.get("vlm_image_quality_notes", ""),
        "raw_text": vlm_layer_raw_result.vlm_layer_raw_text,
    }
    return parsed_fields


def build_vlm_layer_package(vlm_layer_raw_result: VLMRawResult) -> VLMLayerPackage:
    """Build the layer output package required by downstream layers."""
    normalized_result = normalize_vlm_result(vlm_layer_raw_result)
    return VLMLayerPackage(
        vlm_layer_track_id=vlm_layer_raw_result.vlm_layer_track_id,
        vlm_layer_query_type=vlm_layer_raw_result.vlm_layer_query_type,
        vlm_layer_label=str(normalized_result["vlm_layer_label"]),
        vlm_layer_attributes=dict(normalized_result["vlm_layer_attributes"]),
        vlm_layer_confidence=normalized_result["vlm_layer_confidence"],
        vlm_layer_model_id=vlm_layer_raw_result.vlm_layer_model_id,
    )


def build_vlm_ack_package_from_result(vlm_layer_raw_result: VLMRawResult) -> VLMAckPackage:
    normalized_result = normalize_vlm_result(vlm_layer_raw_result)
    retry_reasons = list(normalized_result.get("vlm_retry_reasons", []))
    reason = ", ".join(retry_reasons) if retry_reasons else str(normalized_result.get("vlm_image_quality_notes", ""))
    if not reason:
        reason = str(normalized_result.get("vlm_ack_status", "retry_requested"))
    ack_status = str(normalized_result.get("vlm_ack_status", "retry_requested"))
    return build_vlm_ack_package(
        vlm_ack_track_id=vlm_layer_raw_result.vlm_layer_track_id,
        vlm_ack_status=ack_status,
        vlm_ack_reason=reason,
        vlm_ack_retry_requested=(ack_status == "retry_requested"),
    )


def prepare_vlm_prompt(
    vlm_layer_query_type: str,
    vlm_frame_cropper_layer_package: VLMFrameCropperLayerPackage,
) -> str:
    """Prepare the semantic query or prompt for one crop."""
    if vlm_layer_query_type == "vehicle_semantics_v1":
        allowed_reasons = ", ".join(sorted(_ALLOWED_VLM_RETRY_REASONS))
        return (
            "Look at this cropped vehicle image and respond with JSON only. "
            "First decide whether this image shows a truck-sized vehicle that should stay in the truck pipeline. "
            "If it is not a truck, set is_truck to false, set ack_status to accepted, set retry_reasons to an empty array, and set truck_type to not_truck. "
            "If it is a truck, then decide whether this image is good enough to classify the truck subtype. "
            "If the image is usable, set ack_status to accepted and retry_reasons to an empty array. "
            "If the image is not usable enough, set ack_status to retry_requested and choose one or more retry_reasons from: "
            f"{allowed_reasons}. "
            "Use image_quality_notes for a short human-readable explanation. "
            "Return exactly this JSON schema: "
            '{"ack_status":"accepted or retry_requested",'
            '"is_truck":true,'
            '"retry_reasons":["reason"],'
            '"image_quality_notes":"short note",'
            '"truck_type":"normalized subtype or unknown",'
            '"wheel_count":"integer or unknown",'
            '"estimated_weight_kg":"number, range, or unknown",'
            '"confidence":0.0}'
        )
    if vlm_layer_query_type == "vehicle_semantics_single_shot_v1":
        return (
            "Look at this cropped vehicle image and respond with JSON only. "
            "First decide whether this image shows a truck-sized vehicle. "
            "If it is not a truck, set is_truck to false and truck_type to not_truck. "
            "If it is a truck, set is_truck to true and fill the remaining fields as best you can. "
            "Do your best with the current image and do not ask for another image. "
            "Return exactly this JSON schema: "
            "{\"is_truck\":true,\"truck_type\":\"normalized subtype or unknown\",\"wheel_count\":\"integer or unknown\",\"estimated_weight_kg\":\"number, range, or unknown\",\"confidence\":0.0}"
        )
    if vlm_layer_query_type == "vehicle_class_only_v1":
        return "What type of vehicle is shown in this crop? Answer with one short label only."
    raise ValueError(f"Unsupported vlm_layer_query_type: {vlm_layer_query_type}")


def infer_vlm_semantics(
    vlm_runtime_state: VLMRuntimeState,
    vlm_frame_cropper_layer_package: VLMFrameCropperLayerPackage,
    vlm_prompt_text: str,
) -> str:
    """Run the model on the crop image and return raw decoded text."""
    image = _coerce_image(vlm_frame_cropper_layer_package.vlm_frame_cropper_layer_image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": vlm_prompt_text},
            ],
        }
    ]

    processor = vlm_runtime_state.vlm_runtime_processor
    prompt_text = _apply_chat_template(processor=processor, messages=messages)
    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    )
    inputs = _move_inputs_to_device(
        inputs=inputs,
        device=vlm_runtime_state.vlm_runtime_device,
    )

    with vlm_runtime_state.vlm_runtime_torch.inference_mode():
        generated_ids = vlm_runtime_state.vlm_runtime_model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,
        )

    prompt_tokens = int(inputs["input_ids"].shape[1])
    new_token_ids = generated_ids[:, prompt_tokens:]
    return processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def parse_vlm_response(vlm_layer_raw_text: str) -> dict[str, Any]:
    """Extract structured semantic fields from the model response."""
    stripped_text = vlm_layer_raw_text.strip()
    if not stripped_text:
        return _default_retry_parse("empty_response")

    if stripped_text.startswith("{") and stripped_text.endswith("}"):
        try:
            parsed_json = json.loads(stripped_text)
            return _normalize_json_payload(parsed_json, stripped_text)
        except json.JSONDecodeError:
            pass

    parsed_pairs: dict[str, Any] = {}
    for line in stripped_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed_pairs[key.strip().lower()] = value.strip()

    ack_status = _normalize_ack_status(parsed_pairs.get("ack_status", parsed_pairs.get("status", "retry_requested")))
    retry_reasons = _normalize_retry_reasons(
        parsed_pairs.get("retry_reasons", parsed_pairs.get("retry_reason", "occluded"))
    )
    image_quality_notes = str(parsed_pairs.get("image_quality_notes", parsed_pairs.get("notes", ""))).strip()
    is_truck = _coerce_optional_bool(parsed_pairs.get("is_truck", parsed_pairs.get("truck", "true")))
    truck_type = str(parsed_pairs.get("truck_type", stripped_text.splitlines()[0].strip() if stripped_text else "unknown")).strip() or "unknown"
    wheel_count = parsed_pairs.get("wheel_count", "unknown")
    estimated_weight_kg = parsed_pairs.get("estimated_weight_kg", "unknown")
    confidence = _coerce_optional_float(parsed_pairs.get("confidence"))

    return {
        "is_truck": True if is_truck is None else is_truck,
        "truck_type": truck_type,
        "wheel_count": wheel_count,
        "estimated_weight_kg": estimated_weight_kg,
        "vlm_layer_label": truck_type,
        "vlm_layer_confidence": confidence,
        "vlm_ack_status": ack_status,
        "vlm_retry_reasons": retry_reasons,
        "vlm_image_quality_notes": image_quality_notes,
    }


def serialize_vlm_layer_package(vlm_layer_package: VLMLayerPackage) -> dict[str, Any]:
    """Helper for downstream layers that want a plain dict package."""
    return asdict(vlm_layer_package)


def build_vlm_ack_package(
    vlm_ack_track_id: str,
    vlm_ack_status: str,
    vlm_ack_reason: str,
    vlm_ack_retry_requested: bool = False,
) -> VLMAckPackage:
    """Build a downstream acknowledgement package for cropper/state orchestration."""
    if vlm_ack_status not in _ALLOWED_VLM_ACK_STATUSES:
        raise ValueError(f"vlm_ack_status must be one of {sorted(_ALLOWED_VLM_ACK_STATUSES)}.")
    return VLMAckPackage(
        vlm_ack_track_id=str(vlm_ack_track_id),
        vlm_ack_status=vlm_ack_status,
        vlm_ack_reason=str(vlm_ack_reason),
        vlm_ack_retry_requested=bool(vlm_ack_retry_requested),
    )


def serialize_vlm_ack_package(vlm_ack_package: VLMAckPackage) -> dict[str, Any]:
    return asdict(vlm_ack_package)


def _normalize_json_payload(parsed_json: Any, raw_text: str) -> dict[str, Any]:
    if not isinstance(parsed_json, dict):
        return _default_retry_parse("non_dict_json")
    ack_status = _normalize_ack_status(parsed_json.get("ack_status", "retry_requested"))
    retry_reasons = _normalize_retry_reasons(parsed_json.get("retry_reasons", []))
    image_quality_notes = str(parsed_json.get("image_quality_notes", "")).strip()
    is_truck = _coerce_optional_bool(parsed_json.get("is_truck", True))
    truck_type = str(parsed_json.get("truck_type", "unknown")).strip() or "unknown"
    wheel_count = parsed_json.get("wheel_count", "unknown")
    estimated_weight_kg = parsed_json.get("estimated_weight_kg", "unknown")
    confidence = _coerce_optional_float(parsed_json.get("confidence"))
    return {
        "is_truck": True if is_truck is None else is_truck,
        "truck_type": truck_type,
        "wheel_count": wheel_count,
        "estimated_weight_kg": estimated_weight_kg,
        "vlm_layer_label": truck_type,
        "vlm_layer_confidence": confidence,
        "vlm_ack_status": ack_status,
        "vlm_retry_reasons": retry_reasons,
        "vlm_image_quality_notes": image_quality_notes,
        "raw_text": raw_text,
    }


def _default_retry_parse(reason: str) -> dict[str, Any]:
    return {
        "is_truck": True,
        "truck_type": "unknown",
        "wheel_count": "unknown",
        "estimated_weight_kg": "unknown",
        "vlm_layer_label": "unknown",
        "vlm_layer_confidence": None,
        "vlm_ack_status": "retry_requested",
        "vlm_retry_reasons": ["occluded"],
        "vlm_image_quality_notes": reason,
    }


def _normalize_ack_status(raw_status: Any) -> str:
    value = str(raw_status or "retry_requested").strip().lower()
    if value not in {"accepted", "retry_requested"}:
        return "retry_requested"
    return value


def _normalize_retry_reasons(raw_reasons: Any) -> list[str]:
    if raw_reasons in (None, "", []):
        return []
    if isinstance(raw_reasons, str):
        candidates = [part.strip().lower() for part in raw_reasons.replace(";", ",").split(",")]
    elif isinstance(raw_reasons, list):
        candidates = [str(item).strip().lower() for item in raw_reasons]
    else:
        candidates = [str(raw_reasons).strip().lower()]
    normalized = [reason for reason in candidates if reason in _ALLOWED_VLM_RETRY_REASONS]
    return normalized if normalized else ["occluded"]


def _coerce_optional_float(value: Any) -> float | None:
    if value in (None, "", "unknown"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_bool(value: Any) -> bool | None:
    if value in (None, "", "unknown"):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return True
    if text in {"false", "no", "n", "0"}:
        return False
    return None


def _resolve_device(torch: Any, requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config_device='cuda' was requested, but CUDA is unavailable.")
    return requested_device


def _read_checkpoint_model_type(model_path: Path) -> str | None:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("model_type") if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _transformers_major_version() -> int:
    import transformers

    first = transformers.__version__.split(".", 1)[0].strip()
    digits = "".join(ch for ch in first if ch.isdigit())
    return int(digits) if digits else 0


def _maybe_require_newer_transformers_for_checkpoint(model_path: Path) -> None:
    """Qwen3.5 checkpoints use model_type qwen3_5, registered in transformers v5+."""
    if _read_checkpoint_model_type(model_path) != "qwen3_5":
        return
    if _transformers_major_version() >= 5:
        return
    import transformers

    raise RuntimeError(
        "This VLM checkpoint is Qwen3.5 (model_type qwen3_5) and requires transformers>=5.0.0. "
        f"Installed transformers is {transformers.__version__}. "
        f"Upgrade with: {_TRANSFORMERS_UPGRADE_HINT}"
    )


def _raise_if_unsupported_qwen35_model_type(exc: BaseException) -> None:
    message = str(exc).lower()
    if "qwen3_5" in message or "does not recognize this architecture" in message:
        import transformers

        raise RuntimeError(
            "Loading this checkpoint failed because the installed `transformers` build does not "
            "include the Qwen3.5 (`qwen3_5`) architecture. Install transformers 5.x or newer. "
            f"Current version: {transformers.__version__}. "
            f"Upgrade with: {_TRANSFORMERS_UPGRADE_HINT}"
        ) from exc


def _load_model_class(transformers: Any) -> Any:
    candidate_names = [
        "AutoModelForImageTextToText",
        "Qwen3_5ForConditionalGeneration",
        "AutoModelForVision2Seq",
    ]
    for candidate_name in candidate_names:
        candidate_model_class = getattr(transformers, candidate_name, None)
        if candidate_model_class is not None:
            return candidate_model_class
    raise RuntimeError(
        "No compatible VLM model class was found in the installed transformers package."
    )


def _coerce_image(vlm_frame_cropper_layer_image: Image.Image | Any) -> Image.Image:
    if isinstance(vlm_frame_cropper_layer_image, Image.Image):
        return vlm_frame_cropper_layer_image.convert("RGB")

    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None

    if np is not None and isinstance(vlm_frame_cropper_layer_image, np.ndarray):
        return Image.fromarray(vlm_frame_cropper_layer_image).convert("RGB")

    raise TypeError(
        "vlm_frame_cropper_layer_image must be a PIL image or numpy ndarray."
    )


def preview_vlm_applied_prompt(vlm_runtime_state: VLMRuntimeState, vlm_prompt_text: str) -> str:
    """Return the chat-template-expanded prompt string (for debug / visualization)."""
    if not vlm_runtime_state.config_vlm_enabled:
        return ""
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": vlm_prompt_text},
            ],
        }
    ]
    return _apply_chat_template(
        processor=vlm_runtime_state.vlm_runtime_processor,
        messages=messages,
    )


def _apply_chat_template(processor: Any, messages: list[dict[str, Any]]) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _move_inputs_to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    moved_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        moved_inputs[key] = value.to(device) if hasattr(value, "to") else value
    return moved_inputs


__all__ = [
    "DEFAULT_VLM_MODEL_PATH",
    "DEFAULT_VLM_QUERY_TYPE",
    "VLMConfig",
    "VLMFrameCropperLayerPackage",
    "VLMLayerPackage",
    "VLMAckPackage",
    "VLMRawResult",
    "VLMRuntimeState",
    "build_vlm_layer_package",
    "build_vlm_ack_package",
    "build_vlm_ack_package_from_result",
    "infer_vlm_semantics",
    "initialize_vlm_layer",
    "normalize_vlm_result",
    "parse_vlm_response",
    "prepare_vlm_prompt",
    "preview_vlm_applied_prompt",
    "run_vlm_inference",
    "serialize_vlm_layer_package",
    "serialize_vlm_ack_package",
]
