from __future__ import annotations

import argparse
import base64
import sys
import time
import uuid
from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from engine import DEFAULT_MODEL_DIR, QwenImageEngine


WEB_APP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen3.5-0.8B Local Chat</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1117;
      --panel: #121b24;
      --panel-2: #182432;
      --line: #2a3a4c;
      --text: #eef5ff;
      --muted: #9db1c7;
      --accent: #5ad1a4;
      --accent-2: #84b8ff;
      --danger: #ff7c7c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "IBM Plex Sans", sans-serif;
      background: radial-gradient(circle at top, #142231 0%, var(--bg) 55%);
      color: var(--text);
    }
    .wrap {
      width: min(1100px, 100%);
      margin: 0 auto;
      padding: 24px 16px 40px;
    }
    .hero { display: grid; gap: 8px; margin-bottom: 18px; }
    .hero h1 { margin: 0; font-size: clamp(28px, 5vw, 42px); line-height: 1.05; }
    .hero p { margin: 0; color: var(--muted); max-width: 800px; }
    .grid { display: grid; grid-template-columns: 340px 1fr; gap: 16px; }
    .card {
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 20px 45px rgba(0,0,0,0.25);
    }
    .card .head { padding: 14px 16px; border-bottom: 1px solid var(--line); background: rgba(255,255,255,0.02); font-weight: 600; }
    .card .body { padding: 16px; }
    label { display: block; margin: 0 0 8px; color: var(--muted); font-size: 13px; letter-spacing: 0.02em; text-transform: uppercase; }
    textarea, input[type="number"], input[type="file"] {
      width: 100%; border: 1px solid var(--line); background: var(--panel-2); color: var(--text); border-radius: 12px; padding: 12px 14px; font: inherit;
    }
    textarea { min-height: 140px; resize: vertical; }
    .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px; }
    .row { margin-bottom: 14px; }
    .toggle { display: flex; align-items: center; gap: 10px; margin-top: 6px; color: var(--muted); font-size: 14px; }
    .toggle input { transform: translateY(1px); }
    .actions { display: flex; gap: 10px; margin-top: 16px; }
    button { border: 0; border-radius: 999px; padding: 12px 18px; font: inherit; font-weight: 700; cursor: pointer; }
    .primary { background: linear-gradient(90deg, var(--accent), var(--accent-2)); color: #051018; flex: 1; }
    .secondary { background: transparent; color: var(--muted); border: 1px solid var(--line); }
    .chat { display: grid; grid-template-rows: auto 1fr; min-height: 680px; }
    .status { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 14px 16px; border-bottom: 1px solid var(--line); color: var(--muted); font-size: 14px; }
    .status strong { color: var(--text); }
    .messages { padding: 18px; display: grid; gap: 14px; align-content: start; overflow: auto; background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00)); }
    .msg { border: 1px solid var(--line); border-radius: 16px; padding: 14px 16px; background: rgba(255,255,255,0.02); white-space: pre-wrap; word-break: break-word; }
    .msg.user { background: rgba(90, 209, 164, 0.08); border-color: rgba(90, 209, 164, 0.35); }
    .msg.assistant { background: rgba(132, 184, 255, 0.07); border-color: rgba(132, 184, 255, 0.28); }
    .msg.system { color: var(--muted); }
    .meta { margin-bottom: 8px; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
    .preview { max-width: 100%; margin-top: 10px; border-radius: 12px; border: 1px solid var(--line); display: none; }
    .error { color: var(--danger); }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } .chat { min-height: 520px; } }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Qwen3.5-0.8B Local Web Chat</h1>
      <p>Use this to get a fast feel for the model. Type a prompt, optionally attach one image, and send it to the local backend already running on this machine.</p>
    </section>
    <section class="grid">
      <div class="card">
        <div class="head">Prompt</div>
        <div class="body">
          <div class="row">
            <label for="prompt">Message</label>
            <textarea id="prompt" placeholder="Ask a question, summarize text, or describe an attached image."></textarea>
          </div>
          <div class="row">
            <label for="image">Optional Image</label>
            <input id="image" type="file" accept="image/*">
            <img id="preview" class="preview" alt="Selected preview">
          </div>
          <div class="controls">
            <div>
              <label for="maxTokens">Max New Tokens</label>
              <input id="maxTokens" type="number" min="1" max="4096" value="512">
            </div>
            <div>
              <label for="temperature">Temperature</label>
              <input id="temperature" type="number" min="0" max="2" step="0.1" value="0.7">
            </div>
          </div>
          <div class="controls">
            <div>
              <label for="topP">Top P</label>
              <input id="topP" type="number" min="0" max="1" step="0.05" value="0.8">
            </div>
            <div>
              <label>&nbsp;</label>
              <div class="toggle">
                <input id="thinking" type="checkbox">
                <span>Enable thinking</span>
              </div>
            </div>
          </div>
          <div class="actions">
            <button id="send" class="primary">Send</button>
            <button id="clear" class="secondary" type="button">Clear Chat</button>
          </div>
        </div>
      </div>
      <div class="card chat">
        <div class="status">
          <div>Status: <strong id="status">Ready</strong></div>
          <div id="serverMeta">Checking backend...</div>
        </div>
        <div id="messages" class="messages"></div>
      </div>
    </section>
  </div>
  <script>
    const promptEl = document.getElementById('prompt');
    const imageEl = document.getElementById('image');
    const previewEl = document.getElementById('preview');
    const maxTokensEl = document.getElementById('maxTokens');
    const temperatureEl = document.getElementById('temperature');
    const topPEl = document.getElementById('topP');
    const thinkingEl = document.getElementById('thinking');
    const sendEl = document.getElementById('send');
    const clearEl = document.getElementById('clear');
    const messagesEl = document.getElementById('messages');
    const statusEl = document.getElementById('status');
    const serverMetaEl = document.getElementById('serverMeta');

    function addMessage(role, text, extra = {}) {
      const node = document.createElement('div');
      node.className = `msg ${role}`;
      const meta = document.createElement('div');
      meta.className = 'meta';
      meta.textContent = extra.label || role;
      node.appendChild(meta);
      const body = document.createElement('div');
      body.textContent = text;
      node.appendChild(body);
      if (extra.imageSrc) {
        const img = document.createElement('img');
        img.className = 'preview';
        img.style.display = 'block';
        img.src = extra.imageSrc;
        img.alt = 'User image';
        node.appendChild(img);
      }
      messagesEl.appendChild(node);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function setStatus(text, isError = false) {
      statusEl.textContent = text;
      statusEl.className = isError ? 'error' : '';
    }

    function fileToDataUrl(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
    }

    imageEl.addEventListener('change', async () => {
      const file = imageEl.files[0];
      if (!file) {
        previewEl.style.display = 'none';
        previewEl.removeAttribute('src');
        return;
      }
      const dataUrl = await fileToDataUrl(file);
      previewEl.src = dataUrl;
      previewEl.style.display = 'block';
    });

    clearEl.addEventListener('click', () => {
      messagesEl.innerHTML = '';
      promptEl.value = '';
      imageEl.value = '';
      previewEl.style.display = 'none';
      previewEl.removeAttribute('src');
      setStatus('Ready');
    });

    sendEl.addEventListener('click', async () => {
      const prompt = promptEl.value.trim();
      if (!prompt) {
        setStatus('Enter a prompt first.', true);
        return;
      }

      sendEl.disabled = true;
      setStatus('Generating...');

      let imageBase64 = null;
      let previewSrc = null;
      const file = imageEl.files[0];
      if (file) {
        previewSrc = await fileToDataUrl(file);
        imageBase64 = previewSrc;
      }

      addMessage('user', prompt, {
        label: file ? 'User prompt + image' : 'User prompt',
        imageSrc: previewSrc,
      });

      try {
        const response = await fetch('/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            prompt,
            image_base64: imageBase64,
            max_new_tokens: Number(maxTokensEl.value),
            temperature: Number(temperatureEl.value),
            top_p: Number(topPEl.value),
            enable_thinking: thinkingEl.checked,
          }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || 'Request failed.');
        }
        addMessage('assistant', data.text || '(No text returned)', {
          label: `Assistant | ${data.device} | ${data.dtype}`,
        });
        setStatus('Ready');
      } catch (error) {
        addMessage('system', String(error), { label: 'Error' });
        setStatus('Request failed', true);
      } finally {
        sendEl.disabled = false;
      }
    });

    fetch('/metadata')
      .then((r) => r.json())
      .then((data) => {
        serverMetaEl.textContent = `${data.device} | ${data.dtype} | ${data.model_path.split(/[\\/]/).slice(-1)[0]}`;
      })
      .catch(() => {
        serverMetaEl.textContent = 'Backend metadata unavailable';
      });

    addMessage('system', 'Server is ready. Enter a prompt and click Send. Add one image if you want multimodal input.', { label: 'System' });
  </script>
</body>
</html>
"""


class GenerateRequest(BaseModel):
    prompt: str
    image_path: str | None = None
    image_base64: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    enable_thinking: bool = False


class GenerateResponse(BaseModel):
    text: str
    model_path: str
    image_path: str | None
    device: str
    dtype: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class MetadataResponse(BaseModel):
    model_path: str
    device: str
    dtype: str
    default_host: str
    default_port: int


class OpenAIImageUrl(BaseModel):
    url: str


class OpenAIInputTextPart(BaseModel):
    type: str
    text: str


class OpenAIInputImagePart(BaseModel):
    type: str
    image_url: OpenAIImageUrl


class OpenAIMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class OpenAIChatRequest(BaseModel):
    model: str | None = None
    messages: list[OpenAIMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.8
    stream: bool = False


ENGINE: QwenImageEngine | None = None
ENGINE_LOCK = Lock()
APP = FastAPI(title="Local Qwen3.5-0.8B Service")


@APP.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(WEB_APP_HTML)


@APP.get("/health")
def health() -> dict[str, str]:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return {
        "status": "ok",
        "model_path": str(ENGINE.model_path),
        "device": ENGINE.device,
        "dtype": str(ENGINE.dtype),
    }


@APP.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return MetadataResponse(
        model_path=str(ENGINE.model_path),
        device=ENGINE.device,
        dtype=str(ENGINE.dtype),
        default_host="127.0.0.1",
        default_port=8010,
    )


@APP.get("/v1/models")
def list_models() -> dict[str, Any]:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    model_id = ENGINE.model_path.name
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


def _suffix_from_data_url(image_base64: str) -> str:
    if image_base64.startswith("data:image/png"):
        return ".png"
    if image_base64.startswith("data:image/webp"):
        return ".webp"
    return ".jpg"


def _write_temp_image(image_base64: str) -> Path:
    payload = image_base64.split(",", 1)[-1]
    image_bytes = base64.b64decode(payload)
    with NamedTemporaryFile(delete=False, suffix=_suffix_from_data_url(image_base64)) as tmp:
        tmp.write(image_bytes)
        return Path(tmp.name)


def _resolve_openai_message_payload(messages: list[OpenAIMessage]) -> tuple[str, str | None]:
    text_parts: list[str] = []
    image_payload: str | None = None
    for message in messages:
        if message.role != "user":
            continue
        if isinstance(message.content, str):
            text_parts.append(message.content)
            continue
        for part in message.content:
            part_type = part.get("type")
            if part_type in {"text", "input_text"}:
                if part.get("text"):
                    text_parts.append(str(part["text"]))
            elif part_type == "image_url":
                image_url = part.get("image_url", {})
                url = image_url.get("url")
                if url and image_payload is None:
                    image_payload = str(url)
    prompt = "\n\n".join(piece for piece in text_parts if piece.strip()).strip()
    return prompt, image_payload


def _run_generate(
    *,
    prompt: str,
    image_path: Path | None,
    image_base64: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
) -> dict[str, Any]:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    temp_image_path: Path | None = None
    effective_image_path = image_path
    if image_base64:
        try:
            temp_image_path = _write_temp_image(image_base64)
            effective_image_path = temp_image_path
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc

    try:
        with ENGINE_LOCK:
            result = ENGINE.generate(
                prompt=prompt,
                image_path=effective_image_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=enable_thinking,
            )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_image_path is not None:
            with suppress(OSError):
                temp_image_path.unlink()

    return result


@APP.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    result = _run_generate(
        prompt=request.prompt,
        image_path=Path(request.image_path) if request.image_path else None,
        image_base64=request.image_base64,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        enable_thinking=request.enable_thinking,
    )
    return GenerateResponse(**result)


@APP.post("/v1/chat/completions")
def chat_completions(request: OpenAIChatRequest) -> dict[str, Any]:
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true is not implemented in this local service.")

    prompt, image_payload = _resolve_openai_message_payload(request.messages)
    if not prompt:
        raise HTTPException(status_code=400, detail="At least one user text message is required.")

    image_path: Path | None = None
    image_base64: str | None = None
    if image_payload:
        if image_payload.startswith("data:image/"):
            image_base64 = image_payload
        elif image_payload.startswith("file://"):
            image_path = Path(image_payload[7:])
        else:
            image_path = Path(image_payload)

    max_new_tokens = request.max_completion_tokens or request.max_tokens or 512
    result = _run_generate(
        prompt=prompt,
        image_path=image_path,
        image_base64=image_base64,
        max_new_tokens=max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        enable_thinking=False,
    )

    model_id = request.model or Path(result["model_path"]).name
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["text"],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Qwen3.5-0.8B inference service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    return parser.parse_args()


def main() -> None:
    global ENGINE
    args = parse_args()
    print("Loading model...", file=sys.stderr, flush=True)
    ENGINE = QwenImageEngine(model_path=args.model_path, device=args.device, dtype=args.dtype)
    print(
        f"Web chat ready at http://{args.host}:{args.port}/ using device={ENGINE.device} dtype={ENGINE.dtype}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"OpenAI-compatible endpoint ready at http://{args.host}:{args.port}/v1/chat/completions",
        file=sys.stderr,
        flush=True,
    )
    uvicorn.run(APP, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
