from __future__ import annotations

import json
import importlib.util
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from backends import (
    gemini_e2b,
    huggingface_local,
    resolve_vlm_backend_name,
    resolve_vlm_backend_runtime_kind,
)


LAYER_DIR = Path(__file__).resolve().parent
REPO_ROOT = LAYER_DIR.parent.parent
YOLO_CLASS_MAP_PATH = REPO_ROOT / "src" / "yolo-layer" / "class_map.py"
DEFAULT_VLM_MODEL_PATH = LAYER_DIR / "Qwen3.5-0.8B"
DEFAULT_VLM_QUERY_TYPE = "vehicle_semantics_v1"
DEFAULT_VLM_DEBUG_OUTPUT_DIR = Path(
    r"E:\OneDrive\desktop\01_2026_Projects\01_2026_Cornell_26_Spring\MAE_4221_IoT\DesignProject\Evaluations\07-sample-json-strs"
)
_TRANSFORMERS_UPGRADE_HINT = (
    'pip install -U "transformers>=5.0.0,<6.0.0"'
)
_ALLOWED_VLM_ACK_STATUSES = {"accepted", "retry_requested", "finalize_with_current"}
_ALLOWED_VLM_RETRY_REASONS = {
    "occluded",
    "bad_angle",
}
_NO_ACK_REASON = "no"


def _set_decoder_only_processor_left_padding(processor: Any) -> None:
    """Causal / decoder-only LMs need left padding when batching variable-length prompts."""
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "padding_side"):
        tok.padding_side = "left"


@dataclass(slots=True)
class VLMConfig:
    config_vlm_enabled: bool = False
    config_vlm_backend: str = "auto"
    config_vlm_model: str = str(DEFAULT_VLM_MODEL_PATH)
    config_vlm_api_key_env: str = "GEMINI_API_KEY"
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
    config_vlm_backend: str
    config_vlm_model: str
    config_vlm_api_key_env: str
    config_device: str
    vlm_runtime_backend_kind: str
    vlm_runtime_device: str
    vlm_runtime_dtype: str
    vlm_runtime_model_id: str
    vlm_runtime_backend_state: dict[str, Any] = field(default_factory=dict)
    vlm_runtime_processor: Any = None
    vlm_runtime_model: Any = None
    vlm_runtime_torch: Any = None


def initialize_vlm_layer(config: VLMConfig) -> VLMRuntimeState:
    """Load the configured VLM and return a runtime state package."""
    if not config.config_vlm_enabled:
        return VLMRuntimeState(
            config_vlm_enabled=False,
            config_vlm_backend=str(config.config_vlm_backend or "auto"),
            config_vlm_model=config.config_vlm_model,
            config_vlm_api_key_env=str(config.config_vlm_api_key_env or "GEMINI_API_KEY"),
            config_device=config.config_device,
            vlm_runtime_backend_kind="disabled",
            vlm_runtime_device="disabled",
            vlm_runtime_dtype="disabled",
            vlm_runtime_model_id=Path(config.config_vlm_model).name or "disabled",
        )

    resolved_backend_name = resolve_vlm_backend_name(
        config_vlm_backend=config.config_vlm_backend,
        config_vlm_model=config.config_vlm_model,
    )
    runtime_backend_kind = resolve_vlm_backend_runtime_kind(
        config_vlm_backend=config.config_vlm_backend,
        config_vlm_model=config.config_vlm_model,
    )
    if runtime_backend_kind == "huggingface_local":
        backend_state = huggingface_local.initialize_backend(
            backend_name=resolved_backend_name,
            model_path=config.config_vlm_model,
            requested_device=config.config_device,
        )
    elif runtime_backend_kind == "gemini_e2b":
        backend_state = gemini_e2b.initialize_backend(
            model_name=config.config_vlm_model,
            api_key_env=config.config_vlm_api_key_env,
        )
    else:
        raise NotImplementedError(
            f"VLM backend '{resolved_backend_name}' is configured but not implemented yet in this branch."
        )

    return VLMRuntimeState(
        config_vlm_enabled=True,
        config_vlm_backend=resolved_backend_name,
        config_vlm_model=str(config.config_vlm_model),
        config_vlm_api_key_env=str(config.config_vlm_api_key_env or "GEMINI_API_KEY"),
        config_device=config.config_device,
        vlm_runtime_backend_kind=runtime_backend_kind,
        vlm_runtime_device=str(backend_state.get("runtime_device", "remote")),
        vlm_runtime_dtype=str(backend_state.get("runtime_dtype", "remote")),
        vlm_runtime_model_id=str(backend_state.get("model_id", config.config_vlm_model)),
        vlm_runtime_backend_state=dict(backend_state),
        vlm_runtime_processor=backend_state.get("processor"),
        vlm_runtime_model=backend_state.get("model"),
        vlm_runtime_torch=backend_state.get("torch"),
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


def run_vlm_inference_batch(
    vlm_runtime_state: VLMRuntimeState,
    vlm_frame_cropper_layer_packages: list[VLMFrameCropperLayerPackage],
    vlm_layer_query_types: list[str] | None = None,
) -> list[VLMRawResult]:
    """Run semantic inference on a batch of crops and return raw results.

    Notes:
    - Batching amortizes processor/model overhead and typically improves GPU throughput.
    - If per-item query types are not provided, all items use DEFAULT_VLM_QUERY_TYPE.
    """
    if not vlm_runtime_state.config_vlm_enabled:
        raise RuntimeError("The VLM layer is disabled, so inference cannot run.")
    if not vlm_frame_cropper_layer_packages:
        return []

    query_types = (
        list(vlm_layer_query_types)
        if vlm_layer_query_types is not None
        else [DEFAULT_VLM_QUERY_TYPE] * len(vlm_frame_cropper_layer_packages)
    )
    if len(query_types) != len(vlm_frame_cropper_layer_packages):
        raise ValueError("vlm_layer_query_types must match vlm_frame_cropper_layer_packages length.")

    prompts = [
        prepare_vlm_prompt(vlm_layer_query_type=qt, vlm_frame_cropper_layer_package=pkg)
        for qt, pkg in zip(query_types, vlm_frame_cropper_layer_packages, strict=True)
    ]
    raw_texts = infer_vlm_semantics_batch(
        vlm_runtime_state=vlm_runtime_state,
        vlm_frame_cropper_layer_packages=vlm_frame_cropper_layer_packages,
        vlm_prompt_texts=prompts,
    )

    model_id = vlm_runtime_state.vlm_runtime_model_id
    return [
        VLMRawResult(
            vlm_layer_track_id=pkg.vlm_frame_cropper_layer_track_id,
            vlm_layer_query_type=qt,
            vlm_layer_model_id=model_id,
            vlm_layer_raw_text=raw_text,
            vlm_layer_raw_response={"prompt_text": prompt, "batch_index": index},
        )
        for index, (pkg, qt, prompt, raw_text) in enumerate(
            zip(vlm_frame_cropper_layer_packages, query_types, prompts, raw_texts, strict=True)
        )
    ]


def normalize_vlm_result(vlm_layer_raw_result: VLMRawResult) -> dict[str, Any]:
    """Convert the raw model output into stable normalized semantic fields."""
    parsed_fields = parse_vlm_response(vlm_layer_raw_result.vlm_layer_raw_text)
    parsed_fields.setdefault("is_truck", True)
    parsed_fields.setdefault("wheel_count", 0)
    parsed_fields.setdefault("estimated_weight_kg", 0)
    default_ack_status = "accepted" if vlm_layer_raw_result.vlm_layer_query_type == "vehicle_semantics_single_shot_v1" else "retry_requested"
    default_retry_reasons = [] if default_ack_status == "accepted" else ["occluded"]
    parsed_fields.setdefault("vlm_ack_status", default_ack_status)
    parsed_fields.setdefault("vlm_retry_reasons", default_retry_reasons)
    parsed_fields.setdefault("vlm_layer_confidence", None)

    if parsed_fields.get("is_truck") is False:
        parsed_fields["vlm_ack_status"] = "accepted"
        parsed_fields["vlm_retry_reasons"] = []

    parsed_fields["vlm_layer_label"] = _NO_ACK_REASON if parsed_fields.get("is_truck") is False else "target_label_match"
    parsed_fields["vlm_layer_attributes"] = {
        "is_truck": parsed_fields.get("is_truck", True),
        "wheel_count": parsed_fields.get("wheel_count", 0),
        "estimated_weight_kg": parsed_fields.get("estimated_weight_kg", 0),
        "vlm_ack_status": parsed_fields.get("vlm_ack_status", "retry_requested"),
        "vlm_retry_reasons": list(parsed_fields.get("vlm_retry_reasons", [])),
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
    if normalized_result.get("is_truck") is False:
        return build_vlm_ack_package(
            vlm_ack_track_id=vlm_layer_raw_result.vlm_layer_track_id,
            vlm_ack_status="accepted",
            vlm_ack_reason=_NO_ACK_REASON,
            vlm_ack_retry_requested=False,
        )
    if vlm_layer_raw_result.vlm_layer_query_type == "vehicle_semantics_single_shot_v1":
        return build_vlm_ack_package(
            vlm_ack_track_id=vlm_layer_raw_result.vlm_layer_track_id,
            vlm_ack_status="accepted",
            vlm_ack_reason="single_shot_accepted",
            vlm_ack_retry_requested=False,
        )
    retry_reasons = list(normalized_result.get("vlm_retry_reasons", []))
    reason = ", ".join(retry_reasons)
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
    allowed_labels = _load_supported_detector_labels()
    allowed_label_text = ", ".join(allowed_labels)
    json_only_prompt = (
        f"Allowed: {allowed_label_text}\n"
        "is_truck=no OR one JSON only with keys is_truck,wheel_count,estimated_weight_kg,ack_status,retry_reasons "
        '(ack_status accepted|retry_requested). Ex: {"is_truck":true,"wheel_count":6,"estimated_weight_kg":9000,'
        '"ack_status":"accepted","retry_reasons":[]}'
    )
    if vlm_layer_query_type == "vehicle_semantics_v1":
        return json_only_prompt
    if vlm_layer_query_type == "vehicle_semantics_single_shot_v1":
        return json_only_prompt
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
    if vlm_runtime_state.vlm_runtime_backend_kind == "huggingface_local":
        return huggingface_local.infer_single(
            backend_state=vlm_runtime_state.vlm_runtime_backend_state,
            image=image,
            prompt_text=vlm_prompt_text,
        )
    if vlm_runtime_state.vlm_runtime_backend_kind == "gemini_e2b":
        return gemini_e2b.infer_single(
            backend_state=vlm_runtime_state.vlm_runtime_backend_state,
            image=image,
            prompt_text=vlm_prompt_text,
        )
    raise RuntimeError(
        f"Unsupported VLM runtime backend kind: {vlm_runtime_state.vlm_runtime_backend_kind}"
    )


def infer_vlm_semantics_batch(
    vlm_runtime_state: VLMRuntimeState,
    vlm_frame_cropper_layer_packages: list[VLMFrameCropperLayerPackage],
    vlm_prompt_texts: list[str],
) -> list[str]:
    """Run the model on a batch of crop images and return raw decoded texts (one per crop)."""
    if not vlm_frame_cropper_layer_packages:
        return []
    if len(vlm_prompt_texts) != len(vlm_frame_cropper_layer_packages):
        raise ValueError("vlm_prompt_texts must match vlm_frame_cropper_layer_packages length.")

    images = [
        _coerce_image(pkg.vlm_frame_cropper_layer_image)
        for pkg in vlm_frame_cropper_layer_packages
    ]
    if vlm_runtime_state.vlm_runtime_backend_kind == "huggingface_local":
        return huggingface_local.infer_batch(
            backend_state=vlm_runtime_state.vlm_runtime_backend_state,
            images=images,
            prompt_texts=vlm_prompt_texts,
        )
    if vlm_runtime_state.vlm_runtime_backend_kind == "gemini_e2b":
        return gemini_e2b.infer_batch(
            backend_state=vlm_runtime_state.vlm_runtime_backend_state,
            images=images,
            prompt_texts=vlm_prompt_texts,
        )
    raise RuntimeError(
        f"Unsupported VLM runtime backend kind: {vlm_runtime_state.vlm_runtime_backend_kind}"
    )


def parse_vlm_response(vlm_layer_raw_text: str) -> dict[str, Any]:
    """Extract structured semantic fields from the model response."""
    stripped_text = vlm_layer_raw_text.strip()
    if not stripped_text:
        return _default_retry_parse("empty_response")

    lowered = stripped_text.lower()
    normalized_no_text = lowered.replace(" ", "")
    if (
        lowered == "no"
        or lowered.startswith("no\n")
        or normalized_no_text == "is_truck=no"
        or normalized_no_text.startswith("is_truck=no\n")
        or normalized_no_text == '{"is_truck":false}'
    ):
        return {
            "is_truck": False,
            "wheel_count": 0,
            "estimated_weight_kg": 0,
            "vlm_layer_label": _NO_ACK_REASON,
            "vlm_layer_confidence": None,
            "vlm_ack_status": "accepted",
            "vlm_retry_reasons": [],
        }
    json_start = stripped_text.find("{")
    json_end = stripped_text.rfind("}")
    if json_start != -1 and json_end > json_start:
        try:
            parsed_json = json.loads(stripped_text[json_start:json_end + 1])
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
    is_truck = _coerce_optional_bool(parsed_pairs.get("is_truck", parsed_pairs.get("truck", "true")))
    wheel_count = _coerce_int_value(parsed_pairs.get("wheel_count"), default=0)
    estimated_weight_kg = _coerce_int_value(parsed_pairs.get("estimated_weight_kg"), default=0)

    return {
        "is_truck": True if is_truck is None else is_truck,
        "wheel_count": wheel_count,
        "estimated_weight_kg": estimated_weight_kg,
        "vlm_layer_label": _NO_ACK_REASON if (True if is_truck is None else is_truck) is False else "target_label_match",
        "vlm_layer_confidence": None,
        "vlm_ack_status": ack_status,
        "vlm_retry_reasons": retry_reasons,
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


def build_vlm_output_json(
    vlm_layer_raw_result: VLMRawResult,
    include_raw_result: bool = True,
) -> dict[str, Any]:
    """Build one downstream-friendly JSON payload from raw, normalized, layer, and ack outputs."""
    normalized_result = normalize_vlm_result(vlm_layer_raw_result)
    vlm_layer_package = build_vlm_layer_package(vlm_layer_raw_result)
    vlm_ack_package = build_vlm_ack_package_from_result(vlm_layer_raw_result)

    output_payload = {
        "vlm_layer_output_type": "vlm_output_json_v1",
        "vlm_layer_track_id": vlm_layer_raw_result.vlm_layer_track_id,
        "vlm_layer_query_type": vlm_layer_raw_result.vlm_layer_query_type,
        "vlm_layer_model_id": vlm_layer_raw_result.vlm_layer_model_id,
        "normalized_result": normalized_result,
        "vlm_layer_package": serialize_vlm_layer_package(vlm_layer_package),
        "vlm_ack_package": serialize_vlm_ack_package(vlm_ack_package),
    }
    if include_raw_result:
        output_payload["vlm_raw_result"] = {
            "vlm_layer_track_id": vlm_layer_raw_result.vlm_layer_track_id,
            "vlm_layer_query_type": vlm_layer_raw_result.vlm_layer_query_type,
            "vlm_layer_model_id": vlm_layer_raw_result.vlm_layer_model_id,
            "vlm_layer_raw_text": vlm_layer_raw_result.vlm_layer_raw_text,
            "vlm_layer_raw_response": dict(vlm_layer_raw_result.vlm_layer_raw_response),
        }
    return output_payload


def format_vlm_output_json(
    vlm_layer_raw_result: VLMRawResult,
    indent: int = 2,
    include_raw_result: bool = True,
) -> str:
    """Return the combined VLM output payload as a JSON string."""
    return json.dumps(
        build_vlm_output_json(
            vlm_layer_raw_result=vlm_layer_raw_result,
            include_raw_result=include_raw_result,
        ),
        indent=indent,
    )


def build_sample_vlm_output_json_strings() -> list[str]:
    """Return representative VLM output JSON strings for terminal demos and quick validation."""
    sample_results = [
        VLMRawResult(
            vlm_layer_track_id="sample-track-accepted",
            vlm_layer_query_type="vehicle_semantics_v1",
            vlm_layer_model_id="demo-qwen",
            vlm_layer_raw_text=(
                '{"is_truck":true,"wheel_count":18,"estimated_weight_kg":24000,'
                '"ack_status":"accepted","retry_reasons":[]}'
            ),
            vlm_layer_raw_response={"prompt_text": "demo accepted prompt"},
        ),
        VLMRawResult(
            vlm_layer_track_id="sample-track-retry",
            vlm_layer_query_type="vehicle_semantics_v1",
            vlm_layer_model_id="demo-qwen",
            vlm_layer_raw_text=(
                '{"is_truck":true,"wheel_count":0,"estimated_weight_kg":0,'
                '"ack_status":"retry_requested","retry_reasons":["occluded"]}'
            ),
            vlm_layer_raw_response={"prompt_text": "demo retry prompt"},
        ),
        VLMRawResult(
            vlm_layer_track_id="sample-track-no",
            vlm_layer_query_type="vehicle_semantics_single_shot_v1",
            vlm_layer_model_id="demo-qwen",
            vlm_layer_raw_text="is_truck=no",
            vlm_layer_raw_response={"prompt_text": "demo no prompt"},
        ),
    ]
    return [format_vlm_output_json(sample_result) for sample_result in sample_results]


def save_vlm_debug_image(
    vlm_frame_cropper_layer_package: VLMFrameCropperLayerPackage,
    vlm_layer_raw_result: VLMRawResult,
    output_dir: str | Path = DEFAULT_VLM_DEBUG_OUTPUT_DIR,
    file_stem: str | None = None,
) -> Path:
    """Save a debug image with the crop plus prompt/output text and return the saved path."""
    crop_image = _coerce_image(vlm_frame_cropper_layer_package.vlm_frame_cropper_layer_image)
    prompt_text = str(vlm_layer_raw_result.vlm_layer_raw_response.get("prompt_text", "")).strip()
    output_text = vlm_layer_raw_result.vlm_layer_raw_text.strip()

    metadata_items = _build_vlm_debug_metadata_items(vlm_layer_raw_result=vlm_layer_raw_result)
    prompt_lines = _wrap_debug_text(prompt_text, width=42)
    output_lines = _wrap_debug_text(output_text, width=42)
    debug_image = _render_vlm_debug_image(
        crop_image=crop_image,
        metadata_items=metadata_items,
        prompt_lines=prompt_lines,
        output_lines=output_lines,
    )

    target_dir = Path(output_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    if file_stem is None:
        file_stem = (
            f"{_sanitize_filename_fragment(vlm_layer_raw_result.vlm_layer_track_id)}"
            f"_{_sanitize_filename_fragment(vlm_layer_raw_result.vlm_layer_query_type)}"
        )
    output_path = target_dir / f"{file_stem}.png"
    debug_image.save(output_path)
    return output_path


def save_sample_vlm_output_debug_images(
    sample_image: Image.Image | Any,
    output_dir: str | Path = DEFAULT_VLM_DEBUG_OUTPUT_DIR,
) -> list[Path]:
    """Save debug images for the built-in sample VLM outputs."""
    sample_results = [
        VLMRawResult(
            vlm_layer_track_id="sample-track-accepted",
            vlm_layer_query_type="vehicle_semantics_v1",
            vlm_layer_model_id="demo-qwen",
            vlm_layer_raw_text=(
                '{"is_truck":true,"wheel_count":18,"estimated_weight_kg":24000,'
                '"ack_status":"accepted","retry_reasons":[]}'
            ),
            vlm_layer_raw_response={"prompt_text": "demo accepted prompt"},
        ),
        VLMRawResult(
            vlm_layer_track_id="sample-track-retry",
            vlm_layer_query_type="vehicle_semantics_v1",
            vlm_layer_model_id="demo-qwen",
            vlm_layer_raw_text=(
                '{"is_truck":true,"wheel_count":0,"estimated_weight_kg":0,'
                '"ack_status":"retry_requested","retry_reasons":["occluded"]}'
            ),
            vlm_layer_raw_response={"prompt_text": "demo retry prompt"},
        ),
        VLMRawResult(
            vlm_layer_track_id="sample-track-no",
            vlm_layer_query_type="vehicle_semantics_single_shot_v1",
            vlm_layer_model_id="demo-qwen",
            vlm_layer_raw_text="is_truck=no",
            vlm_layer_raw_response={"prompt_text": "demo no prompt"},
        ),
    ]
    saved_paths: list[Path] = []
    for index, sample_result in enumerate(sample_results, start=1):
        sample_package = VLMFrameCropperLayerPackage(
            vlm_frame_cropper_layer_track_id=sample_result.vlm_layer_track_id,
            vlm_frame_cropper_layer_image=sample_image,
            vlm_frame_cropper_layer_bbox=None,
        )
        saved_paths.append(
            save_vlm_debug_image(
                vlm_frame_cropper_layer_package=sample_package,
                vlm_layer_raw_result=sample_result,
                output_dir=output_dir,
                file_stem=f"{index:02d}_{_sanitize_filename_fragment(sample_result.vlm_layer_track_id)}",
            )
        )
    return saved_paths


def _normalize_json_payload(parsed_json: Any, raw_text: str) -> dict[str, Any]:
    if not isinstance(parsed_json, dict):
        return _default_retry_parse("non_dict_json")
    ack_status = _normalize_ack_status(parsed_json.get("ack_status", "retry_requested"))
    retry_reasons = _normalize_retry_reasons(parsed_json.get("retry_reasons", []))
    is_truck = _coerce_optional_bool(parsed_json.get("is_truck", True))
    wheel_count = _coerce_int_value(parsed_json.get("wheel_count"), default=0)
    estimated_weight_kg = _coerce_int_value(parsed_json.get("estimated_weight_kg"), default=0)
    is_target = True if is_truck is None else is_truck
    return {
        "is_truck": is_target,
        "wheel_count": wheel_count,
        "estimated_weight_kg": estimated_weight_kg,
        "vlm_layer_label": _NO_ACK_REASON if is_target is False else "target_label_match",
        "vlm_layer_confidence": None,
        "vlm_ack_status": ack_status,
        "vlm_retry_reasons": retry_reasons,
        "raw_text": raw_text,
    }


def _default_retry_parse(reason: str) -> dict[str, Any]:
    return {
        "is_truck": True,
        "wheel_count": 0,
        "estimated_weight_kg": 0,
        "vlm_layer_label": "unknown",
        "vlm_layer_confidence": None,
        "vlm_ack_status": "retry_requested",
        "vlm_retry_reasons": ["occluded"],
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


def _coerce_int_value(value: Any, default: int = 0) -> int:
    if value in (None, "", "unknown"):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


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
    if vlm_runtime_state.vlm_runtime_backend_kind == "gemini_e2b":
        return vlm_prompt_text
    if vlm_runtime_state.vlm_runtime_backend_kind == "huggingface_local":
        processor = vlm_runtime_state.vlm_runtime_backend_state.get("processor")
        if processor is not None:
            return huggingface_local.apply_prompt_template(
                processor=processor,
                prompt_text=vlm_prompt_text,
            )
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


def _load_supported_detector_labels() -> list[str]:
    try:
        spec = importlib.util.spec_from_file_location("vlm_prompt_class_map", YOLO_CLASS_MAP_PATH)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load class_map spec.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target_classes = getattr(module, "TARGET_CLASSES", {})
        labels = [str(label).strip() for label in target_classes.values() if str(label).strip()]
        deduped = sorted(dict.fromkeys(labels))
        return deduped or ["truck"]
    except Exception:
        return ["truck"]


def _build_vlm_debug_text_lines(
    vlm_layer_raw_result: VLMRawResult,
    prompt_text: str,
    output_text: str,
) -> list[str]:
    body_sections = [
        ("Track ID", vlm_layer_raw_result.vlm_layer_track_id),
        ("Query Type", vlm_layer_raw_result.vlm_layer_query_type),
        ("Model ID", vlm_layer_raw_result.vlm_layer_model_id),
        ("Prompt", prompt_text or "(empty)"),
        ("VLM Output", output_text or "(empty)"),
    ]
    lines = ["VLM Debug Output", ""]
    for heading, content in body_sections:
        lines.append(f"{heading}:")
        wrapped = _wrap_debug_text(str(content), width=52)
        lines.extend(wrapped)
        lines.append("")
    return lines[:-1] if lines and not lines[-1] else lines


def _build_vlm_debug_metadata_items(vlm_layer_raw_result: VLMRawResult) -> list[tuple[str, str]]:
    return [
        ("Track ID", str(vlm_layer_raw_result.vlm_layer_track_id)),
        ("Model ID", str(vlm_layer_raw_result.vlm_layer_model_id)),
    ]


def _render_vlm_debug_image(
    crop_image: Image.Image,
    metadata_items: list[tuple[str, str]],
    prompt_lines: list[str],
    output_lines: list[str],
) -> Image.Image:
    margin = 32
    panel_gap = 24
    top_card_width = 1800
    background_color = (18, 18, 20)
    text_color = (235, 235, 240)
    muted_text_color = (170, 170, 180)
    border_color = (90, 90, 100)
    panel_color = (28, 28, 32)
    title_font = _load_debug_font(38)
    section_font = _load_debug_font(28)
    body_font = _load_debug_font(22)

    crop_max_width = 1800
    crop_max_height = 1400
    preview_image = crop_image.copy()
    preview_image.thumbnail((crop_max_width, crop_max_height))

    body_line_height = _measure_line_height(body_font) + 10
    metadata_row_height = body_line_height + 18
    top_card_height = max(340, preview_image.height + 72)
    text_column_width = (top_card_width - panel_gap) // 2
    prompt_card_height = 90 + (len(prompt_lines) * body_line_height)
    output_card_height = 90 + (len(output_lines) * body_line_height)
    lower_cards_height = max(prompt_card_height, output_card_height)
    content_height = top_card_height + panel_gap + lower_cards_height
    canvas_height = content_height + (margin * 2)
    canvas_width = top_card_width + (margin * 2)

    canvas = Image.new("RGB", (canvas_width, canvas_height), background_color)
    draw = ImageDraw.Draw(canvas)

    top_x = margin
    top_y = margin
    draw.rounded_rectangle(
        [top_x, top_y, top_x + top_card_width, top_y + top_card_height],
        radius=18,
        fill=panel_color,
        outline=border_color,
        width=2,
    )
    draw.text((top_x + 20, top_y + 18), "VLM Result", fill=text_color, font=title_font)

    metadata_x = top_x + 24
    metadata_y = top_y + 92
    for label, value in metadata_items:
        draw.text((metadata_x, metadata_y), label, fill=muted_text_color, font=body_font)
        draw.text((metadata_x + 280, metadata_y), value, fill=text_color, font=body_font)
        metadata_y += metadata_row_height

    image_area_margin = 20
    image_box_height = top_card_height - 40
    image_box_width = min(860, top_card_width // 2)
    resampling_module = getattr(Image, "Resampling", Image)
    scale = max((image_box_width - 12) / max(1, preview_image.width), (image_box_height - 12) / max(1, preview_image.height))
    scaled_size = (
        max(1, int(preview_image.width * scale)),
        max(1, int(preview_image.height * scale)),
    )
    scaled_preview = preview_image.resize(scaled_size, resampling_module.BICUBIC)
    image_box_x = top_x + top_card_width - image_box_width - image_area_margin
    image_box_y = top_y + (top_card_height - image_box_height) // 2
    draw.rounded_rectangle(
        [image_box_x, image_box_y, image_box_x + image_box_width, image_box_y + image_box_height],
        radius=14,
        fill=(22, 22, 26),
        outline=border_color,
        width=2,
    )
    crop_left = max(0, (scaled_preview.width - image_box_width) // 2)
    crop_top = max(0, (scaled_preview.height - image_box_height) // 2)
    cropped_preview = scaled_preview.crop(
        (
            crop_left,
            crop_top,
            min(scaled_preview.width, crop_left + image_box_width),
            min(scaled_preview.height, crop_top + image_box_height),
        )
    )
    canvas.paste(cropped_preview, (image_box_x, image_box_y))

    lower_y = top_y + top_card_height + panel_gap
    prompt_x = top_x
    output_x = top_x + text_column_width + panel_gap
    _draw_debug_text_card(
        draw=draw,
        x=prompt_x,
        y=lower_y,
        width=text_column_width,
        height=lower_cards_height,
        title="Prompt Sent To VLM",
        lines=prompt_lines,
        title_font=section_font,
        body_font=body_font,
        panel_color=panel_color,
        border_color=border_color,
        text_color=text_color,
    )
    _draw_debug_text_card(
        draw=draw,
        x=output_x,
        y=lower_y,
        width=text_column_width,
        height=lower_cards_height,
        title="Actual VLM Output",
        lines=output_lines,
        title_font=section_font,
        body_font=body_font,
        panel_color=panel_color,
        border_color=border_color,
        text_color=text_color,
    )

    return canvas


def _sanitize_filename_fragment(value: Any) -> str:
    text = str(value).strip().replace(" ", "_")
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return sanitized.strip("_") or "vlm"


def _wrap_debug_text(text: str, width: int) -> list[str]:
    formatted_text = _pretty_format_debug_text(text)
    wrapped_lines: list[str] = []
    for raw_line in formatted_text.splitlines() or [""]:
        if not raw_line:
            wrapped_lines.append("")
            continue
        normalized_line = raw_line.replace(",", ", ").replace(":{", ": {")
        wrapped_lines.extend(
            textwrap.wrap(
                normalized_line,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=True,
            )
            or [normalized_line]
        )
    return wrapped_lines or ["(empty)"]


def _pretty_format_debug_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "(empty)"
    json_start = stripped.find("{")
    json_end = stripped.rfind("}")
    if json_start == -1 or json_end <= json_start:
        return stripped
    prefix = stripped[:json_start].rstrip()
    json_blob = stripped[json_start:json_end + 1]
    suffix = stripped[json_end + 1:].strip()
    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError:
        return stripped
    pretty_json = json.dumps(parsed, indent=2)
    segments = [segment for segment in [prefix, pretty_json, suffix] if segment]
    return "\n".join(segments)


def _measure_line_height(font: ImageFont.ImageFont | ImageFont.FreeTypeFont) -> int:
    try:
        bbox = font.getbbox("Ag")
        return max(1, bbox[3] - bbox[1])
    except AttributeError:
        return 12


def _load_debug_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    font_candidates = [
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_debug_text_card(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    lines: list[str],
    title_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    body_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    panel_color: tuple[int, int, int],
    border_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> None:
    line_height = _measure_line_height(body_font) + 10
    draw.rounded_rectangle(
        [x, y, x + width, y + height],
        radius=18,
        fill=panel_color,
        outline=border_color,
        width=2,
    )
    draw.text((x + 20, y + 18), title, fill=text_color, font=title_font)
    text_y = y + 66
    for line in lines:
        draw.text((x + 22, text_y), line, fill=text_color, font=body_font)
        text_y += line_height


__all__ = [
    "DEFAULT_VLM_MODEL_PATH",
    "DEFAULT_VLM_DEBUG_OUTPUT_DIR",
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
    "build_vlm_output_json",
    "build_sample_vlm_output_json_strings",
    "infer_vlm_semantics",
    "infer_vlm_semantics_batch",
    "format_vlm_output_json",
    "initialize_vlm_layer",
    "normalize_vlm_result",
    "parse_vlm_response",
    "prepare_vlm_prompt",
    "preview_vlm_applied_prompt",
    "run_vlm_inference",
    "run_vlm_inference_batch",
    "save_sample_vlm_output_debug_images",
    "save_vlm_debug_image",
    "serialize_vlm_layer_package",
    "serialize_vlm_ack_package",
]


if __name__ == "__main__":
    for sample_json in build_sample_vlm_output_json_strings():
        print(sample_json)
        print()
