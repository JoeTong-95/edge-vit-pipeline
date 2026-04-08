from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


LAYER_DIR = Path(__file__).resolve().parent
DEFAULT_VLM_MODEL_PATH = LAYER_DIR / "Qwen3.5-0.8B"
DEFAULT_VLM_QUERY_TYPE = "vehicle_semantics_v1"


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

    runtime_device = _resolve_device(torch=torch, requested_device=config.config_device)
    runtime_dtype = torch.float16 if runtime_device == "cuda" else torch.float32

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
    parsed_fields.setdefault("vlm_layer_label", parsed_fields.get("truck_type", "unknown"))
    parsed_fields.setdefault("vlm_layer_attributes", {})
    parsed_fields["vlm_layer_attributes"].update(
        {
            "truck_type": parsed_fields.get("truck_type", "unknown"),
            "wheel_count": parsed_fields.get("wheel_count", "unknown"),
            "estimated_weight_kg": parsed_fields.get("estimated_weight_kg", "unknown"),
            "raw_text": vlm_layer_raw_result.vlm_layer_raw_text,
        }
    )
    parsed_fields.setdefault("vlm_layer_confidence", None)
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


def prepare_vlm_prompt(
    vlm_layer_query_type: str,
    vlm_frame_cropper_layer_package: VLMFrameCropperLayerPackage,
) -> str:
    """Prepare the semantic query or prompt for one crop."""
    if vlm_layer_query_type == "vehicle_semantics_v1":
        return (
            "Look at this cropped vehicle image and answer with exactly these keys and no extra text:\n"
            "truck_type: <short answer>\n"
            "wheel_count: <integer or unknown>\n"
            "estimated_weight_kg: <number or range or unknown>"
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
            max_new_tokens=128,
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
        return {
            "truck_type": "unknown",
            "wheel_count": "unknown",
            "estimated_weight_kg": "unknown",
            "vlm_layer_label": "unknown",
            "vlm_layer_attributes": {},
            "vlm_layer_confidence": None,
        }

    with_json_braces = stripped_text
    if stripped_text.startswith("{") and stripped_text.endswith("}"):
        try:
            parsed_json = json.loads(stripped_text)
            return {
                "truck_type": parsed_json.get("truck_type", "unknown"),
                "wheel_count": parsed_json.get("wheel_count", "unknown"),
                "estimated_weight_kg": parsed_json.get("estimated_weight_kg", "unknown"),
                "vlm_layer_label": parsed_json.get("truck_type", "unknown"),
                "vlm_layer_attributes": {
                    key: value for key, value in parsed_json.items() if key != "truck_type"
                },
                "vlm_layer_confidence": parsed_json.get("confidence"),
            }
        except json.JSONDecodeError:
            with_json_braces = stripped_text

    parsed_pairs: dict[str, Any] = {}
    for line in with_json_braces.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed_pairs[key.strip().lower()] = value.strip()

    truck_type = parsed_pairs.get("truck_type", stripped_text.splitlines()[0].strip())
    wheel_count = parsed_pairs.get("wheel_count", "unknown")
    estimated_weight_kg = parsed_pairs.get("estimated_weight_kg", "unknown")

    return {
        "truck_type": truck_type,
        "wheel_count": wheel_count,
        "estimated_weight_kg": estimated_weight_kg,
        "vlm_layer_label": truck_type,
        "vlm_layer_attributes": {
            "wheel_count": wheel_count,
            "estimated_weight_kg": estimated_weight_kg,
        },
        "vlm_layer_confidence": None,
    }


def serialize_vlm_layer_package(vlm_layer_package: VLMLayerPackage) -> dict[str, Any]:
    """Helper for downstream layers that want a plain dict package."""
    return asdict(vlm_layer_package)


def _resolve_device(torch: Any, requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config_device='cuda' was requested, but CUDA is unavailable.")
    return requested_device


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
    "VLMRawResult",
    "VLMRuntimeState",
    "build_vlm_layer_package",
    "infer_vlm_semantics",
    "initialize_vlm_layer",
    "normalize_vlm_result",
    "parse_vlm_response",
    "prepare_vlm_prompt",
    "run_vlm_inference",
    "serialize_vlm_layer_package",
]
