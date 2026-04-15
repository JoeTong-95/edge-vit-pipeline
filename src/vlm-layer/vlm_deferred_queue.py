from __future__ import annotations

import base64
import json
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True, slots=True)
class DeferredVLMTask:
    track_id: str
    dispatch_frame_id: int
    query_type: str
    prompt_text: str
    crop_png_base64: str
    bbox: tuple[int, int, int, int] | None = None
    created_at_unix_s: float | None = None


_FILE_LOCKS: dict[str, threading.Lock] = {}


def append_deferred_task(path: str | Path, task: DeferredVLMTask) -> None:
    """Append one task as JSONL to `path` (creates parent dirs)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = _FILE_LOCKS.setdefault(str(target.resolve()), threading.Lock())
    payload = {
        "track_id": task.track_id,
        "dispatch_frame_id": int(task.dispatch_frame_id),
        "query_type": str(task.query_type),
        "prompt_text": str(task.prompt_text),
        "crop_png_base64": str(task.crop_png_base64),
        "bbox": list(task.bbox) if task.bbox is not None else None,
        "created_at_unix_s": float(task.created_at_unix_s) if task.created_at_unix_s is not None else None,
    }
    line = json.dumps(payload, ensure_ascii=False)
    with lock:
        with open(target, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_deferred_tasks(path: str | Path, limit: int | None = None) -> list[DeferredVLMTask]:
    source = Path(path)
    if not source.is_file():
        return []
    tasks: list[DeferredVLMTask] = []
    with open(source, encoding="utf-8") as f:
        for raw_line in f:
            if limit is not None and len(tasks) >= int(limit):
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            tasks.append(_task_from_json(data))
    return tasks


def decode_crop_image(crop_png_base64: str) -> Image.Image:
    raw = base64.b64decode(crop_png_base64.encode("utf-8"))
    image = Image.open(BytesIO(raw)).convert("RGB")
    return image


def encode_crop_image_to_png_base64(image_or_array: Image.Image | Any) -> str:
    """Encode a crop image (PIL or numpy array) as PNG base64 for JSONL persistence."""
    if isinstance(image_or_array, Image.Image):
        image = image_or_array.convert("RGB")
    else:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise TypeError("Non-PIL crops require numpy installed.") from exc
        if isinstance(image_or_array, np.ndarray):
            image = Image.fromarray(image_or_array).convert("RGB")
        else:
            raise TypeError("crop must be a PIL.Image or numpy ndarray.")

    buf = BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _task_from_json(data: dict[str, Any]) -> DeferredVLMTask:
    bbox = data.get("bbox")
    return DeferredVLMTask(
        track_id=str(data.get("track_id", "")),
        dispatch_frame_id=int(data.get("dispatch_frame_id", -1)),
        query_type=str(data.get("query_type", "")),
        prompt_text=str(data.get("prompt_text", "")),
        crop_png_base64=str(data.get("crop_png_base64", "")),
        bbox=tuple(int(x) for x in bbox) if isinstance(bbox, list) and len(bbox) == 4 else None,
        created_at_unix_s=float(data["created_at_unix_s"]) if data.get("created_at_unix_s") is not None else None,
    )


__all__ = [
    "DeferredVLMTask",
    "append_deferred_task",
    "decode_crop_image",
    "encode_crop_image_to_png_base64",
    "load_deferred_tasks",
]

