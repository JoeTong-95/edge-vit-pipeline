from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_TRANSFORMERS_UPGRADE_HINT = 'pip install -U "transformers>=5.0.0,<6.0.0"'


def initialize_backend(
    *,
    backend_name: str,
    model_path: str,
    requested_device: str,
) -> dict[str, Any]:
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

    resolved_model_path = Path(model_path).expanduser().resolve()
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Configured VLM model path was not found: {resolved_model_path}")
    _raise_if_git_lfs_pointer_checkpoint(resolved_model_path)

    _maybe_require_newer_transformers_for_checkpoint(model_path=resolved_model_path)

    runtime_device = _resolve_device(torch=torch, requested_device=requested_device)
    runtime_dtype = torch.bfloat16 if runtime_device == "cuda" else torch.float32

    model_name_lower = resolved_model_path.name.strip().lower()
    is_qwen_like = ("qwen" in model_name_lower) or (backend_name == "qwen_0_8b")

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "torch_dtype": runtime_dtype,
    }
    if is_qwen_like:
        # Qwen loads can spike memory on Jetson; this reduces peak host memory.
        load_kwargs["low_cpu_mem_usage"] = True
    if runtime_device == "cuda":
        # Prefer low-peak loading on GPU machines; fallback below if accelerate/device_map is unavailable.
        load_kwargs["low_cpu_mem_usage"] = True
        load_kwargs["device_map"] = "auto"

    try:
        processor = AutoProcessor.from_pretrained(
            resolved_model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        _set_decoder_only_processor_left_padding(processor)
        model_class = _load_model_class(transformers)
        model = model_class.from_pretrained(resolved_model_path, **load_kwargs)
    except (ImportError, RuntimeError) as exc:
        _raise_if_jetson_cuda_runtime_failure(
            exc=exc,
            runtime_device=runtime_device,
            model_id=resolved_model_path.name,
            stage="load_model",
        )
        if load_kwargs.get("device_map") == "auto" and _should_retry_without_device_map(exc):
            # Retry only when accelerate/device_map support is missing.
            load_kwargs.pop("device_map", None)
            model = model_class.from_pretrained(resolved_model_path, **load_kwargs)
        else:
            raise
    except (KeyError, ValueError) as exc:
        _raise_if_unsupported_qwen35_model_type(exc)
        raise

    if not hasattr(model, "hf_device_map"):
        try:
            model = model.to(runtime_device)
        except RuntimeError as exc:
            _raise_if_jetson_cuda_move_failure(
                exc=exc,
                model_path=resolved_model_path,
                runtime_device=runtime_device,
                runtime_dtype=runtime_dtype,
            )
            raise
    model.eval()

    return {
        "backend_name": backend_name,
        "model_path": str(resolved_model_path),
        "runtime_device": runtime_device,
        "runtime_dtype": str(runtime_dtype),
        "model_id": resolved_model_path.name,
        "processor": processor,
        "model": model,
        "torch": torch,
    }


def infer_single(
    *,
    backend_state: dict[str, Any],
    image: Any,
    prompt_text: str,
) -> str:
    processor = backend_state["processor"]
    prompt = apply_prompt_template(processor=processor, prompt_text=prompt_text)
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
    )
    inputs = _move_inputs_to_device(
        inputs=inputs,
        device=str(backend_state["runtime_device"]),
    )
    torch = backend_state["torch"]
    model = backend_state["model"]
    try:
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
            )
    except RuntimeError as exc:
        _raise_if_jetson_cuda_runtime_failure(
            exc=exc,
            runtime_device=str(backend_state["runtime_device"]),
            model_id=str(backend_state.get("model_id", "unknown")),
            stage="generate",
        )
        raise

    prompt_tokens = int(inputs["input_ids"].shape[1])
    new_token_ids = generated_ids[:, prompt_tokens:]
    return processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def infer_batch(
    *,
    backend_state: dict[str, Any],
    images: list[Any],
    prompt_texts: list[str],
) -> list[str]:
    if not images:
        return []
    if len(images) != len(prompt_texts):
        raise ValueError("images and prompt_texts must match in length.")

    processor = backend_state["processor"]
    prompt_batch = [
        apply_prompt_template(processor=processor, prompt_text=prompt_text)
        for prompt_text in prompt_texts
    ]
    inputs = processor(
        text=prompt_batch,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = _move_inputs_to_device(
        inputs=inputs,
        device=str(backend_state["runtime_device"]),
    )
    torch = backend_state["torch"]
    model = backend_state["model"]
    try:
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
            )
    except RuntimeError as exc:
        _raise_if_jetson_cuda_runtime_failure(
            exc=exc,
            runtime_device=str(backend_state["runtime_device"]),
            model_id=str(backend_state.get("model_id", "unknown")),
            stage="generate_batch",
        )
        raise

    input_ids = inputs.get("input_ids")
    if input_ids is None:
        decoded = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [text.strip() for text in decoded]

    outputs: list[str] = []
    for row_index in range(int(generated_ids.shape[0])):
        prompt_tokens = int(input_ids[row_index].shape[0])
        new_token_ids = generated_ids[row_index:row_index + 1, prompt_tokens:]
        decoded = processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        outputs.append(decoded)
    return outputs


def apply_prompt_template(*, processor: Any, prompt_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
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


def _set_decoder_only_processor_left_padding(processor: Any) -> None:
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "padding_side"):
        tok.padding_side = "left"


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


def _looks_like_git_lfs_pointer(file_path: Path) -> bool:
    if not file_path.is_file():
        return False
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(200)
    except OSError:
        return False
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def _raise_if_git_lfs_pointer_checkpoint(model_path: Path) -> None:
    pointer_candidates = [
        model_path / "tokenizer.json",
        model_path / "model.safetensors",
        model_path / "model.safetensors-00001-of-00001.safetensors",
    ]
    flagged = [path.name for path in pointer_candidates if _looks_like_git_lfs_pointer(path)]
    if not flagged:
        return
    raise RuntimeError(
        "This local VLM checkpoint is not fully downloaded; Git LFS pointer files were found for: "
        + ", ".join(flagged)
        + ". Hydrate the checkpoint files on this machine before using this backend."
    )


def _transformers_major_version() -> int:
    import transformers

    first = transformers.__version__.split(".", 1)[0].strip()
    digits = "".join(ch for ch in first if ch.isdigit())
    return int(digits) if digits else 0


def _maybe_require_newer_transformers_for_checkpoint(model_path: Path) -> None:
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


def _move_inputs_to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    moved_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        moved_inputs[key] = value.to(device) if hasattr(value, "to") else value
    return moved_inputs


def _should_retry_without_device_map(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        hint in message
        for hint in (
            "accelerate",
            "device_map",
            "dispatch_model",
            "requires accelerate",
        )
    )


def _raise_if_jetson_cuda_move_failure(
    *,
    exc: BaseException,
    model_path: Path,
    runtime_device: str,
    runtime_dtype: Any,
) -> None:
    if runtime_device != "cuda":
        return

    message = str(exc)
    lowered = message.lower()
    if "cudacachingallocator" not in lowered and "nvml_success == r" not in lowered:
        return

    raise RuntimeError(
        "Moving this Hugging Face VLM checkpoint onto CUDA failed during full-model placement. "
        f"model={model_path.name} dtype={runtime_dtype}. "
        "On this Jetson, that usually means the model cannot be placed fully on GPU memory with the "
        "current loader path. The previous automatic retry without device_map can hide the original "
        "failure and lead to a harder allocator crash, so the backend now stops here and reports the "
        "CUDA placement failure directly."
    ) from exc


def _raise_if_jetson_cuda_runtime_failure(
    *,
    exc: BaseException,
    runtime_device: str,
    model_id: str,
    stage: str,
) -> None:
    if runtime_device != "cuda":
        return

    message = str(exc)
    lowered = message.lower()
    if not any(
        hint in lowered
        for hint in (
            "cudacachingallocator",
            "nvml_success == r",
            "cublas_status_execution_failed",
            "cuda error",
        )
    ):
        return

    raise RuntimeError(
        "CUDA inference failed for this Hugging Face VLM on Jetson during "
        f"{stage}. model={model_id}. "
        "The switcher is selecting the configured CUDA path, but this backend/device combination is "
        "not completing bounded inference on the current machine. Treat this as a real CUDA runtime "
        "capacity/stability failure rather than a switcher-resolution bug."
    ) from exc
