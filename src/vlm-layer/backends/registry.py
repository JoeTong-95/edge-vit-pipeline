from __future__ import annotations

from pathlib import Path

SUPPORTED_VLM_BACKENDS = {
    "auto",
    "huggingface_local",
    "smolvlm_256m",
    "qwen_0_8b",
    "gemini_e2b",
}

_HUGGINGFACE_LOCAL_BACKENDS = {
    "huggingface_local",
    "smolvlm_256m",
    "qwen_0_8b",
}


def resolve_vlm_backend_name(config_vlm_backend: str, config_vlm_model: str) -> str:
    requested = str(config_vlm_backend or "auto").strip().lower()
    if requested not in SUPPORTED_VLM_BACKENDS:
        raise ValueError(
            f"config_vlm_backend must be one of {sorted(SUPPORTED_VLM_BACKENDS)}."
        )
    if requested != "auto":
        return requested

    model_name = Path(str(config_vlm_model or "")).name.strip().lower()
    if "smolvlm" in model_name:
        return "smolvlm_256m"
    if "qwen" in model_name:
        return "qwen_0_8b"
    if model_name:
        return "huggingface_local"
    return "huggingface_local"


def resolve_vlm_backend_runtime_kind(config_vlm_backend: str, config_vlm_model: str) -> str:
    backend_name = resolve_vlm_backend_name(
        config_vlm_backend=config_vlm_backend,
        config_vlm_model=config_vlm_model,
    )
    if backend_name in _HUGGINGFACE_LOCAL_BACKENDS:
        return "huggingface_local"
    if backend_name == "gemini_e2b":
        return "gemini_e2b"
    raise ValueError(f"Unsupported resolved VLM backend: {backend_name}")
