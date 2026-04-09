from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parents[2]
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "multimodel" / "Qwen3.5-0.8B"


class QwenImageEngine:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_DIR,
        device: str = "auto",
        dtype: str = "auto",
    ) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyTorch is not installed for the current interpreter. Install a CUDA or CPU "
                "build of torch before running this pipeline."
            ) from exc

        try:
            import transformers
            from transformers import AutoProcessor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "transformers is not installed for the current interpreter. Install a recent "
                "main-branch build that supports Qwen3.5 multimodal models."
            ) from exc

        self.torch = torch
        self.transformers = transformers
        self.AutoProcessor = AutoProcessor
        self.model_path = model_path.expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")

        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.processor = self.AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model_class = self._load_model_class()
        self.model = model_class.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, requested: str) -> str:
        if requested == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return requested

    def _resolve_dtype(self, requested: str) -> Any:
        if requested == "auto":
            return self.torch.float16 if self.device == "cuda" else self.torch.float32
        return getattr(self.torch, requested)

    def _load_model_class(self) -> Any:
        candidate_names = [
            "AutoModelForImageTextToText",
            "Qwen3_5ForConditionalGeneration",
            "AutoModelForVision2Seq",
        ]
        for name in candidate_names:
            model_class = getattr(self.transformers, name, None)
            if model_class is not None:
                return model_class

        raise RuntimeError(
            "No compatible Qwen3.5 multimodal model loader was found in the installed "
            "transformers package. Install a recent main-branch build of transformers."
        )

    def _move_inputs_to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def generate(
        self,
        prompt: str,
        image_path: Path | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        enable_thinking: bool = False,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": []}]
        if image_path is not None:
            resolved_image_path = image_path.expanduser().resolve()
            if not resolved_image_path.exists():
                raise FileNotFoundError(f"Image file not found: {resolved_image_path}")
            image = Image.open(resolved_image_path).convert("RGB")
            messages[0]["content"].append({"type": "image"})
            images = [image]
        else:
            resolved_image_path = None
            images = None

        messages[0]["content"].append({"type": "text", "text": prompt})

        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        processor_kwargs: dict[str, Any] = {
            "text": [prompt_text],
            "return_tensors": "pt",
        }
        if images is not None:
            processor_kwargs["images"] = images

        inputs = self.processor(**processor_kwargs)
        inputs = self._move_inputs_to_device(inputs)

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)

        prompt_tokens = int(inputs["input_ids"].shape[1])
        new_token_ids = generated_ids[:, prompt_tokens:]
        completion_tokens = int(new_token_ids.shape[1])
        output_text = self.processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return {
            "text": output_text,
            "model_path": str(self.model_path),
            "image_path": str(resolved_image_path) if resolved_image_path else None,
            "device": self.device,
            "dtype": str(self.dtype),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
