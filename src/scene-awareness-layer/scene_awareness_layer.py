"""
Scene Awareness Layer (Layer 11)

Implements a lightweight, dependency-optional stub for full-frame scene tagging.

Public functions (per pipeline contract):
- initialize_scene_awareness_layer
- run_scene_awareness_inference
- build_scene_awareness_layer_package
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Optional, Tuple


try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore

try:
    import cv2 as _cv2  # type: ignore
except Exception:  # pragma: no cover
    _cv2 = None  # type: ignore


@dataclass(frozen=True)
class _SceneMetrics:
    brightness_0_1: float
    contrast_0_1: float
    edge_density_0_1: float
    colorfulness_0_1: float


def initialize_scene_awareness_layer(
    config_scene_awareness_enabled: bool, config_device: str = "auto"
) -> dict:
    """
    Prepare runtime state for the scene awareness layer.

    Since this is a stub, there is no model to load. We store enough state
    to preserve a stable interface and allow future replacement with a real model.
    """
    enabled = bool(config_scene_awareness_enabled)
    device = (config_device or "auto").strip().lower()
    if device == "auto":
        device = "cpu"

    return {
        "scene_awareness_runtime_enabled": enabled,
        "scene_awareness_runtime_device": device,
        "scene_awareness_runtime_model_id": "stub_scene_awareness_v1",
        "scene_awareness_runtime_initialized_at": time.time(),
    }


def run_scene_awareness_inference(
    scene_awareness_runtime_state: dict, input_layer_package
) -> dict | None:
    """
    Perform full-frame scene analysis and return a `scene_awareness_layer_package`,
    or None when disabled / unavailable.
    """
    if not isinstance(scene_awareness_runtime_state, dict):
        return None

    if not scene_awareness_runtime_state.get("scene_awareness_runtime_enabled", False):
        return None

    if not isinstance(input_layer_package, dict):
        return None

    image = input_layer_package.get("input_layer_image", None)
    if image is None:
        return None

    metrics = _compute_scene_metrics(image)
    label, confidence, attributes = _label_from_metrics(metrics)

    raw_result = {
        "label": label,
        "attributes": attributes,
        "confidence": confidence,
        "model_id": scene_awareness_runtime_state.get(
            "scene_awareness_runtime_model_id", "stub_scene_awareness_v1"
        ),
        "metrics": {
            "brightness_0_1": metrics.brightness_0_1,
            "contrast_0_1": metrics.contrast_0_1,
            "edge_density_0_1": metrics.edge_density_0_1,
            "colorfulness_0_1": metrics.colorfulness_0_1,
        },
    }

    return build_scene_awareness_layer_package(input_layer_package, raw_result)


def build_scene_awareness_layer_package(input_layer_package, raw_result: dict) -> dict:
    """
    Create `scene_awareness_layer_package` per contract.

    Required fields:
    - scene_awareness_layer_frame_id
    - scene_awareness_layer_timestamp
    - scene_awareness_layer_label
    - scene_awareness_layer_attributes
    - scene_awareness_layer_confidence
    """
    frame_id = None
    timestamp = None
    if isinstance(input_layer_package, dict):
        frame_id = input_layer_package.get("input_layer_frame_id", None)
        timestamp = input_layer_package.get("input_layer_timestamp", None)

    label = None
    attributes = {}
    confidence = None
    if isinstance(raw_result, dict):
        label = raw_result.get("label", None)
        attributes = raw_result.get("attributes", {}) or {}
        confidence = raw_result.get("confidence", None)

        # include extra raw fields in attributes (non-contract, but safe)
        if "model_id" in raw_result:
            attributes = dict(attributes)
            attributes.setdefault("scene_awareness_model_id", raw_result["model_id"])
        if "metrics" in raw_result and isinstance(raw_result["metrics"], dict):
            attributes = dict(attributes)
            attributes.setdefault("scene_awareness_metrics", raw_result["metrics"])

    return {
        "scene_awareness_layer_frame_id": frame_id,
        "scene_awareness_layer_timestamp": timestamp,
        "scene_awareness_layer_label": label,
        "scene_awareness_layer_attributes": attributes,
        "scene_awareness_layer_confidence": confidence,
    }


def _compute_scene_metrics(image: Any) -> _SceneMetrics:
    """
    Compute simple scene metrics with optional numpy/opencv acceleration.

    Accepted image forms (best-effort):
    - numpy array (H,W,3) uint8 BGR/RGB
    - numpy array (H,W) grayscale
    - OpenCV image (same as numpy)
    - nested lists: H x W x 3 or H x W
    """
    if _np is not None:
        arr = _to_numpy(image)
        if arr is not None:
            return _metrics_from_numpy(arr)

    # pure-python fallback for nested lists
    return _metrics_from_python_lists(image)


def _to_numpy(image: Any):
    if _np is None:
        return None
    if isinstance(image, _np.ndarray):
        return image
    try:
        return _np.array(image)
    except Exception:
        return None


def _metrics_from_numpy(arr) -> _SceneMetrics:
    # Normalize shapes
    if arr is None:
        return _SceneMetrics(0.0, 0.0, 0.0, 0.0)

    # If image is float in [0,1], scale to [0,255] for consistent heuristics.
    a = arr
    if getattr(a, "dtype", None) is not None:
        try:
            if str(a.dtype).startswith("float"):
                a = (a * 255.0).clip(0, 255)
        except Exception:
            pass

    # Convert to grayscale for brightness/contrast/edges
    gray = None
    if a.ndim == 2:
        gray = a.astype("float32")
    elif a.ndim == 3 and a.shape[2] >= 3:
        if _cv2 is not None:
            try:
                # Assume BGR (OpenCV default); if it's RGB the metrics are still reasonable.
                gray = _cv2.cvtColor(a.astype("uint8"), _cv2.COLOR_BGR2GRAY).astype(
                    "float32"
                )
            except Exception:
                gray = (
                    0.114 * a[..., 0] + 0.587 * a[..., 1] + 0.299 * a[..., 2]
                ).astype("float32")
        else:
            gray = (0.114 * a[..., 0] + 0.587 * a[..., 1] + 0.299 * a[..., 2]).astype(
                "float32"
            )
    else:
        # Unknown format
        return _SceneMetrics(0.0, 0.0, 0.0, 0.0)

    # Brightness and contrast
    mean = float(gray.mean())
    std = float(gray.std())
    brightness_0_1 = _clamp01(mean / 255.0)
    contrast_0_1 = _clamp01(std / 128.0)  # rough normalization

    # Edge density: use Canny if available; else gradient magnitude threshold
    edge_density_0_1 = 0.0
    try:
        if _cv2 is not None and gray.size > 0:
            g8 = gray.clip(0, 255).astype("uint8")
            edges = _cv2.Canny(g8, 80, 160)
            edge_density_0_1 = float((edges > 0).mean())
        elif gray.size > 0:
            gy = _np.abs(gray[1:, :] - gray[:-1, :])
            gx = _np.abs(gray[:, 1:] - gray[:, :-1])
            mag = 0.5 * (gy.mean() + gx.mean())
            edge_density_0_1 = _clamp01(float(mag) / 32.0)
    except Exception:
        edge_density_0_1 = 0.0

    # Colorfulness: simple channel std proxy
    colorfulness_0_1 = 0.0
    try:
        if a.ndim == 3 and a.shape[2] >= 3:
            a3 = a[..., :3].astype("float32")
            # (std of channels) / 128 normalized
            colorfulness_0_1 = _clamp01(float(a3.std(axis=(0, 1)).mean()) / 128.0)
    except Exception:
        colorfulness_0_1 = 0.0

    return _SceneMetrics(
        brightness_0_1=brightness_0_1,
        contrast_0_1=contrast_0_1,
        edge_density_0_1=_clamp01(edge_density_0_1),
        colorfulness_0_1=colorfulness_0_1,
    )


def _metrics_from_python_lists(image: Any) -> _SceneMetrics:
    # Best-effort: treat as HxW or HxWx3 nested lists of numbers.
    try:
        h = len(image)
        if h == 0:
            return _SceneMetrics(0.0, 0.0, 0.0, 0.0)
        w = len(image[0])
        if w == 0:
            return _SceneMetrics(0.0, 0.0, 0.0, 0.0)
    except Exception:
        return _SceneMetrics(0.0, 0.0, 0.0, 0.0)

    # Sample pixels to limit runtime (pure python)
    step_y = max(1, h // 64)
    step_x = max(1, w // 64)

    gray_vals = []
    color_std_proxy = []
    for y in range(0, h, step_y):
        row = image[y]
        for x in range(0, w, step_x):
            p = row[x]
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                b, g, r = float(p[0]), float(p[1]), float(p[2])
                gray = 0.114 * b + 0.587 * g + 0.299 * r
                gray_vals.append(gray)
                color_std_proxy.append((abs(r - g) + abs(g - b) + abs(b - r)) / 3.0)
            else:
                gray_vals.append(float(p))

    if not gray_vals:
        return _SceneMetrics(0.0, 0.0, 0.0, 0.0)

    mean = sum(gray_vals) / len(gray_vals)
    var = sum((v - mean) ** 2 for v in gray_vals) / len(gray_vals)
    std = math.sqrt(var)

    # Rough edge density: average absolute difference with right/bottom neighbors on the same sample grid
    edge_score = 0.0
    edge_count = 0
    for y in range(0, h - step_y, step_y):
        for x in range(0, w - step_x, step_x):
            v = _sample_gray(image, y, x)
            vr = _sample_gray(image, y, x + step_x)
            vb = _sample_gray(image, y + step_y, x)
            if v is None or vr is None or vb is None:
                continue
            edge_score += abs(v - vr) + abs(v - vb)
            edge_count += 2

    edge_density_0_1 = 0.0
    if edge_count > 0:
        edge_density_0_1 = _clamp01((edge_score / edge_count) / 32.0)

    colorfulness_0_1 = 0.0
    if color_std_proxy:
        colorfulness_0_1 = _clamp01((sum(color_std_proxy) / len(color_std_proxy)) / 64.0)

    return _SceneMetrics(
        brightness_0_1=_clamp01(mean / 255.0),
        contrast_0_1=_clamp01(std / 128.0),
        edge_density_0_1=edge_density_0_1,
        colorfulness_0_1=colorfulness_0_1,
    )


def _sample_gray(image: Any, y: int, x: int) -> Optional[float]:
    try:
        p = image[y][x]
        if isinstance(p, (list, tuple)) and len(p) >= 3:
            b, g, r = float(p[0]), float(p[1]), float(p[2])
            return 0.114 * b + 0.587 * g + 0.299 * r
        return float(p)
    except Exception:
        return None


def _label_from_metrics(metrics: _SceneMetrics) -> Tuple[str, float, Dict[str, Any]]:
    """
    Turn metrics into a simple scene label + confidence.

    Labels are intentionally coarse and model-agnostic:
    - dark_scene / bright_scene
    - low_contrast / high_contrast
    - busy_scene / calm_scene (edge density proxy)
    - colorful_scene / muted_scene
    """
    tags = []

    if metrics.brightness_0_1 < 0.35:
        tags.append("dark_scene")
    elif metrics.brightness_0_1 > 0.65:
        tags.append("bright_scene")
    else:
        tags.append("normal_brightness")

    if metrics.contrast_0_1 < 0.25:
        tags.append("low_contrast")
    elif metrics.contrast_0_1 > 0.55:
        tags.append("high_contrast")
    else:
        tags.append("medium_contrast")

    if metrics.edge_density_0_1 > 0.12:
        tags.append("busy_scene")
    else:
        tags.append("calm_scene")

    if metrics.colorfulness_0_1 > 0.35:
        tags.append("colorful_scene")
    else:
        tags.append("muted_scene")

    label = ",".join(tags)

    # Confidence: higher when metrics are more extreme (farther from midpoints)
    score = 0.0
    score += abs(metrics.brightness_0_1 - 0.5)
    score += abs(metrics.contrast_0_1 - 0.4)
    score += abs(metrics.edge_density_0_1 - 0.08)
    score += abs(metrics.colorfulness_0_1 - 0.25)
    confidence = _clamp01(0.35 + 0.45 * (score / 2.0))

    attributes = {
        "scene_awareness_tags": tags,
        "scene_awareness_stub": True,
    }
    return label, confidence, attributes


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)

