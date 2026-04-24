from __future__ import annotations

import atexit
import base64
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image


_SERVER_PROCESSES: list[subprocess.Popen[str]] = []


def initialize_backend(
    *,
    model_path: str,
    requested_device: str,
) -> dict[str, Any]:
    model_file, mmproj_file = _resolve_model_paths(model_path)
    server_port = _pick_free_port()
    server_url = f"http://127.0.0.1:{server_port}"

    ctx_size = str(os.environ.get("GEMMA_E2B_LOCAL_CTX_SIZE", "1024")).strip() or "1024"
    batch_size = str(os.environ.get("GEMMA_E2B_LOCAL_BATCH_SIZE", "128")).strip() or "128"
    ubatch_size = str(os.environ.get("GEMMA_E2B_LOCAL_UBATCH_SIZE", "32")).strip() or "32"
    command = [
        "/home/jetson/llama.cpp/build/bin/llama-server",
        "-m",
        str(model_file),
        "--mmproj",
        str(mmproj_file),
        "--host",
        "127.0.0.1",
        "--port",
        str(server_port),
        "-c",
        ctx_size,
        "-b",
        batch_size,
        "-ub",
        ubatch_size,
        "--threads",
        "4",
        "--parallel",
        "1",
        "--no-warmup",
    ]
    if str(requested_device).strip().lower() == "cpu":
        command.extend(["--device", "none", "--gpu-layers", "0", "--no-mmproj-offload"])
        runtime_device = "cpu"
    else:
        gpu_layers = str(os.environ.get("GEMMA_E2B_LOCAL_GPU_LAYERS", "4")).strip() or "4"
        command.extend(["--device", "CUDA0", "--gpu-layers", gpu_layers])
        no_mmproj_offload = str(os.environ.get("GEMMA_E2B_LOCAL_NO_MMPROJ_OFFLOAD", "1")).strip().lower()
        if no_mmproj_offload in {"1", "true", "yes"}:
            command.append("--no-mmproj-offload")
        runtime_device = "cuda"

    log_file = tempfile.NamedTemporaryFile(
        mode="w+",
        encoding="utf-8",
        prefix="gemma_e2b_local_",
        suffix=".log",
        delete=False,
    )
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _SERVER_PROCESSES.append(process)
    _wait_for_server_ready(server_url=server_url, process=process, log_path=Path(log_file.name))

    return {
        "model_id": model_file.name,
        "runtime_device": runtime_device,
        "runtime_dtype": "gguf_q4_q8",
        "model_path": str(model_file),
        "mmproj_path": str(mmproj_file),
        "server_url": server_url,
        "server_pid": int(process.pid),
        "server_log_path": str(log_file.name),
    }


def infer_single(
    *,
    backend_state: dict[str, Any],
    image: Any,
    prompt_text: str,
) -> str:
    data_url = _image_to_data_url(image)
    payload = {
        "model": str(backend_state["model_id"]),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0.0,
        "n_predict": 24,
    }
    response = httpx.post(
        f"{backend_state['server_url']}/v1/chat/completions",
        json=payload,
        timeout=180.0,
    )
    response.raise_for_status()
    body = response.json()
    return _extract_chat_completion_text(body).strip()


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


def _resolve_model_paths(model_path: str) -> tuple[Path, Path]:
    candidate = Path(model_path).expanduser().resolve()
    if candidate.is_file():
        if candidate.suffix.lower() != ".gguf":
            raise ValueError(f"gemma_e2b_local expects a GGUF file or a directory, got: {candidate}")
        mmproj_guess = candidate.parent / f"mmproj-{candidate.name}"
        if not mmproj_guess.exists():
            raise FileNotFoundError(
                f"Could not locate mmproj file next to {candidate}. Expected: {mmproj_guess}"
            )
        return candidate, mmproj_guess
    if not candidate.is_dir():
        raise FileNotFoundError(f"Configured gemma_e2b_local path was not found: {candidate}")

    model_files = sorted(
        path for path in candidate.glob("*.gguf") if "mmproj" not in path.name.lower()
    )
    mmproj_files = sorted(
        path for path in candidate.glob("*.gguf") if "mmproj" in path.name.lower()
    )
    if not model_files or not mmproj_files:
        raise FileNotFoundError(
            f"Expected both model and mmproj GGUF files under {candidate}."
        )
    return model_files[0], mmproj_files[0]


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server_ready(server_url: str, process: subprocess.Popen[str], log_path: Path) -> None:
    deadline = time.time() + 90.0
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            try:
                last_error = log_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                last_error = ""
            raise RuntimeError(
                "Local Gemma E2B server exited during startup.\n"
                + last_error[-4000:]
            )
        for endpoint in ("/health", "/v1/models"):
            try:
                response = httpx.get(f"{server_url}{endpoint}", timeout=2.0)
                if response.status_code < 500:
                    return
            except Exception as exc:
                last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(
        f"Timed out waiting for local Gemma E2B server at {server_url}. Last error: {last_error}"
    )


def _image_to_data_url(image: Any) -> str:
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.fromarray(image).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        pil_image.save(temp_path, format="PNG")
        raw_bytes = temp_path.read_bytes()
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_chat_completion_text(response_body: dict[str, Any]) -> str:
    choices = response_body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _shutdown_servers() -> None:
    while _SERVER_PROCESSES:
        process = _SERVER_PROCESSES.pop()
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass


atexit.register(_shutdown_servers)
