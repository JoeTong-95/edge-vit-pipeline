# Qwen3.5-0.8B Multimodal Pipeline

Local image-and-text inference tooling for the model already stored at `models/multimodel/Qwen3.5-0.8B`.

## What this does

- loads the local Qwen3.5-0.8B model from disk
- supports direct one-shot CLI inference
- supports a persistent local backend service that keeps the model loaded
- lets any local terminal send text plus an optional image path to that service
- includes a simple local browser chat page for quickly feeling out the model
- exposes a minimal OpenAI-compatible `/v1/chat/completions` endpoint for benchmarking alignment
- includes a terminal benchmark runner for quick latency and throughput checks

## Files

- `engine.py`: shared model loading and generation logic
- `chat_image.py`: direct local CLI inference entrypoint
- `service.py`: persistent FastAPI backend plus browser chat page at `/`
- `ask.py`: terminal client for talking to the running service
- `benchmark.py`: terminal benchmark runner against the OpenAI-compatible route
- `run-service.bat`: Windows launcher for the backend service
- `run-web-chat.bat`: Windows launcher for the browser chat page
- `run-benchmark.bat`: Windows launcher for terminal benchmark runs
- `ask.bat`: Windows launcher for the terminal client
- repo root `run-win.bat`: launch the service locally from the workspace root
- repo root `run-win-docker.bat`: launch the service in Docker from the workspace root

## Quickest Way To Feel The Model

For a local browser chat window on Windows:

```powershell
pipelines\multimodel\qwen35-0.8b\run-web-chat.bat
```

Then visit:

```text
http://127.0.0.1:8010/
```

## Quick Benchmark In Terminal

Text-only benchmark:

```powershell
pipelines\multimodel\qwen35-0.8b\run-benchmark.bat
```

Multimodal benchmark with one image:

```powershell
pipelines\multimodel\qwen35-0.8b\run-benchmark.bat --image "E:\path\to\image.jpg"
```

This launcher starts the local service automatically if needed, runs warmup plus measured iterations, and prints latency and output tokens-per-second in the terminal.

## Notes

- The service keeps the model loaded in memory, so later requests are much faster than first load.
- The browser UI is intentionally lightweight and meant for quick local testing, not production integration.
- The OpenAI-compatible route is intentionally minimal and focused on compatibility for benchmarking and simple client integration.
- The HTTP service serializes generation calls through a single lock so multiple clients do not race the same loaded model.
- `--dtype auto` uses `float16` on CUDA and `float32` on CPU.
- If the installed `transformers` build is too old, the service will fail early with an upgrade message.
- If `torch.cuda.is_available()` is false, the service will run on CPU.
