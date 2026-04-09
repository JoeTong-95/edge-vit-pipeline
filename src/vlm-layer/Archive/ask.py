from __future__ import annotations

import argparse
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Talk to the local Qwen3.5-0.8B service.")
    parser.add_argument("prompt", help="Prompt to send to the service.")
    parser.add_argument("--image", type=Path, default=None, help="Optional local image path.")
    parser.add_argument("--url", default="http://127.0.0.1:8010/generate", help="Service endpoint URL.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "prompt": args.prompt,
        "image_path": str(args.image.resolve()) if args.image else None,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "enable_thinking": args.enable_thinking,
    }
    response = requests.post(args.url, json=payload, timeout=600)
    response.raise_for_status()
    data = response.json()
    print(f"Device: {data['device']}")
    print(f"Image: {data['image_path']}")
    print("")
    print(data["text"])


if __name__ == "__main__":
    main()
