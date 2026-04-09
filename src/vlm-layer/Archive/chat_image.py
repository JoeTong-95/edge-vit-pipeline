from __future__ import annotations

import argparse
from pathlib import Path

from engine import DEFAULT_MODEL_DIR, QwenImageEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local image-to-text inference with Qwen3.5-0.8B."
    )
    parser.add_argument("image_path", type=Path, nargs="?", help="Optional path to a local image file.")
    parser.add_argument("prompt", help="User prompt to send alongside the optional image.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = QwenImageEngine(model_path=args.model_path, device=args.device, dtype=args.dtype)
    result = engine.generate(
        prompt=args.prompt,
        image_path=args.image_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )
    print(f"Model: {result['model_path']}")
    print(f"Image: {result['image_path']}")
    print(f"Device: {result['device']}")
    print(f"Dtype: {result['dtype']}")
    print("")
    print(result["text"])


if __name__ == "__main__":
    main()
