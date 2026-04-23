from __future__ import annotations

import os
from typing import Any


def initialize_backend(
    *,
    model_name: str,
    api_key_env: str,
) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Gemini backend requires the {api_key_env} environment variable to be set."
        )
    try:
        from google import genai
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Gemini backend requires the official Google GenAI SDK. Install it with: pip install -U google-genai"
        ) from exc

    client = genai.Client(api_key=api_key)
    return {
        "model_id": str(model_name).strip(),
        "api_key_env": str(api_key_env).strip(),
        "client": client,
    }


def infer_single(
    *,
    backend_state: dict[str, Any],
    image: Any,
    prompt_text: str,
) -> str:
    response = backend_state["client"].models.generate_content(
        model=str(backend_state["model_id"]),
        contents=[image, prompt_text],
    )
    return str(getattr(response, "text", "") or "").strip()


def infer_batch(
    *,
    backend_state: dict[str, Any],
    images: list[Any],
    prompt_texts: list[str],
) -> list[str]:
    if len(images) != len(prompt_texts):
        raise ValueError("images and prompt_texts must match in length.")
    return [
        infer_single(
            backend_state=backend_state,
            image=image,
            prompt_text=prompt_text,
        )
        for image, prompt_text in zip(images, prompt_texts, strict=True)
    ]
