from __future__ import annotations

import argparse
import base64
import json
import statistics
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
SERVICE_PATH = SCRIPT_DIR / "service.py"
DEFAULT_URL = "http://127.0.0.1:8010"
DEFAULT_TEXT_PROMPT = (
    "You are helping evaluate a compact multimodal model for edge deployment. "
    "Explain how a warehouse robot fleet could use cameras, local inference, and a lightweight language interface "
    "to detect misplaced inventory, summarize what happened, and suggest one next action for a human operator. "
    "Keep the answer structured with a short overview, three operational considerations, and one concrete example."
)
DEFAULT_IMAGE_PROMPT = (
    "Look at the truck image and answer using exactly these three lines with the same keys and no extra text:\n"
    "truck_type: <short answer>\n"
    "wheel_count: <integer or unknown>\n"
    "estimated_weight_kg: <number or range or unknown>"
)
DEFAULT_IMAGE = SCRIPT_DIR / "truckimage.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the local Qwen3.5-0.8B OpenAI-compatible endpoint.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--prompt", default=None, help="Override the default text prompt.")
    parser.add_argument("--image-prompt", default=DEFAULT_IMAGE_PROMPT, help="Prompt to use for the default image benchmark case.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Local image for multimodal benchmarking.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--start-service", action="store_true", help="Start the local service automatically if it is not already running.")
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--text-only", action="store_true", help="Run only the text benchmark case.")
    parser.add_argument("--image-only", action="store_true", help="Run only the image benchmark case.")
    return parser.parse_args()


def encode_image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(suffix, "image/jpeg")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def is_healthy(base_url: str) -> bool:
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.ok
    except requests.RequestException:
        return False


def wait_for_health(base_url: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if is_healthy(base_url):
            return
        time.sleep(1)
    raise TimeoutError(f"Service did not become healthy within {timeout_seconds} seconds.")


def start_service(base_url: str, timeout_seconds: int) -> subprocess.Popen[str]:
    port = base_url.rsplit(":", 1)[-1]
    command = [sys.executable, str(SERVICE_PATH), "--host", "127.0.0.1", "--port", port]
    process = subprocess.Popen(command, cwd=SCRIPT_DIR, text=True)
    wait_for_health(base_url, timeout_seconds)
    return process


def make_payload(*, prompt: str, image: Path | None, max_tokens: int, temperature: float, top_p: float) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image is not None:
        resolved = image.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Image not found: {resolved}")
        content.append({"type": "image_url", "image_url": {"url": encode_image_to_data_url(resolved)}})
    return {
        "model": "Qwen3.5-0.8B",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }


def run_once(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=1800)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    usage = data.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens", 0))
    tok_per_s = completion_tokens / elapsed if elapsed > 0 else 0.0
    return {
        "latency_s": elapsed,
        "completion_tokens": completion_tokens,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "tokens_per_s": tok_per_s,
        "text": data["choices"][0]["message"]["content"],
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_case(*, case_name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [item["latency_s"] for item in results]
    throughputs = [item["tokens_per_s"] for item in results]
    prompt_tokens = [item["prompt_tokens"] for item in results]
    completion_tokens = [item["completion_tokens"] for item in results]
    return {
        "case": case_name,
        "avg_latency_s": statistics.mean(latencies),
        "p50_latency_s": percentile(latencies, 0.50),
        "p95_latency_s": percentile(latencies, 0.95),
        "avg_tokens_per_s": statistics.mean(throughputs),
        "avg_prompt_tokens": statistics.mean(prompt_tokens),
        "avg_completion_tokens": statistics.mean(completion_tokens),
        "sample_output": results[-1]["text"],
    }


def run_case(*, base_url: str, case_name: str, payload: dict[str, Any], runs: int, warmup_runs: int) -> dict[str, Any]:
    print("", flush=True)
    print(f"=== {case_name} ===", flush=True)
    for index in range(warmup_runs):
        print(f"Warmup {index + 1}/{warmup_runs} ...", flush=True)
        run_once(base_url, payload)
    results: list[dict[str, Any]] = []
    for index in range(runs):
        result = run_once(base_url, payload)
        results.append(result)
        print(
            f"Run {index + 1}/{runs}: latency={result['latency_s']:.2f}s, completion_tokens={result['completion_tokens']}, tokens_per_s={result['tokens_per_s']:.2f}",
            flush=True,
        )
    summary = summarize_case(case_name=case_name, results=results)
    print(f"Average latency: {summary['avg_latency_s']:.2f}s", flush=True)
    print(f"P50 latency: {summary['p50_latency_s']:.2f}s", flush=True)
    print(f"P95 latency: {summary['p95_latency_s']:.2f}s", flush=True)
    print(f"Average output tokens/sec: {summary['avg_tokens_per_s']:.2f}", flush=True)
    print(f"Average prompt tokens: {summary['avg_prompt_tokens']:.1f}", flush=True)
    print(f"Average completion tokens: {summary['avg_completion_tokens']:.1f}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.text_only and args.image_only:
        raise SystemExit("Choose either --text-only or --image-only, not both.")

    service_process: subprocess.Popen[str] | None = None
    if not is_healthy(args.url):
        if not args.start_service:
            raise SystemExit(f"Service is not running at {args.url}. Start it first or rerun with --start-service.")
        print(f"Starting local service at {args.url} ...", flush=True)
        service_process = start_service(args.url, args.startup_timeout)
        print("Service is healthy.", flush=True)

    text_prompt = args.prompt or DEFAULT_TEXT_PROMPT
    run_text = not args.image_only
    run_image = not args.text_only

    print("\n=== Benchmark Config ===", flush=True)
    print(f"Endpoint: {args.url}/v1/chat/completions", flush=True)
    print(f"Runs per case: {args.runs}", flush=True)
    print(f"Warmup runs per case: {args.warmup_runs}", flush=True)
    print(f"Max tokens: {args.max_tokens}", flush=True)
    print(f"Temperature: {args.temperature}", flush=True)
    print(f"Top p: {args.top_p}", flush=True)

    summaries: list[dict[str, Any]] = []
    try:
        if run_text:
            text_payload = make_payload(
                prompt=text_prompt,
                image=None,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            summaries.append(run_case(base_url=args.url, case_name="Mid-size text request", payload=text_payload, runs=args.runs, warmup_runs=args.warmup_runs))

        if run_image:
            image_payload = make_payload(
                prompt=args.image_prompt,
                image=args.image,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            summaries.append(run_case(base_url=args.url, case_name="Image request", payload=image_payload, runs=args.runs, warmup_runs=args.warmup_runs))

        print("\n=== JSON Summary ===", flush=True)
        print(json.dumps({"cases": summaries}, indent=2), flush=True)
    finally:
        if service_process is not None:
            service_process.terminate()
            with suppress(Exception):
                service_process.wait(timeout=10)


if __name__ == "__main__":
    main()
