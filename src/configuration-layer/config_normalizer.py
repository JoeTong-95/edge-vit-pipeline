from __future__ import annotations

from typing import Any, Iterable

from config_defaults import DEFAULT_CONFIG_VALUES
from config_types import ConfigurationLayerConfig


def _normalize_bool(value: Any, config_key: str) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False

    raise TypeError(f"{config_key} must be a bool-compatible value.")


def _normalize_resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        cleaned = value.lower().replace(" ", "")
        if "x" in cleaned:
            width_text, height_text = cleaned.split("x", maxsplit=1)
            return (int(width_text), int(height_text))

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        parts = list(value)
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))

    raise TypeError("config_frame_resolution must be a two-item resolution value.")


def _normalize_yolo_imgsz(value: Any) -> tuple[int, int] | None:
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        parts = list(value)
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    raise TypeError("config_yolo_imgsz must be a [H, W] pair or null.")


def normalize_config(raw_config: dict[str, Any]) -> ConfigurationLayerConfig:
    merged = {**DEFAULT_CONFIG_VALUES, **raw_config}

    return ConfigurationLayerConfig(
        config_device=str(merged["config_device"]).strip().lower(),
        config_input_source=str(merged["config_input_source"]).strip().lower(),
        config_input_path=(
            None if merged["config_input_path"] in {None, ""} else str(merged["config_input_path"])
        ),
        config_frame_resolution=_normalize_resolution(merged["config_frame_resolution"]),
        config_roi_enabled=_normalize_bool(merged["config_roi_enabled"], "config_roi_enabled"),
        config_roi_vehicle_count_threshold=int(merged["config_roi_vehicle_count_threshold"]),
        config_yolo_model=str(merged["config_yolo_model"]).strip(),
        config_yolo_confidence_threshold=float(merged["config_yolo_confidence_threshold"]),
        config_yolo_imgsz=_normalize_yolo_imgsz(merged.get("config_yolo_imgsz")),
        config_vlm_enabled=_normalize_bool(merged["config_vlm_enabled"], "config_vlm_enabled"),
        config_vlm_backend=str(merged.get("config_vlm_backend") or "auto").strip().lower(),
        config_vlm_model=str(merged["config_vlm_model"]).strip(),
        config_vlm_api_key_env=str(merged.get("config_vlm_api_key_env") or "GEMINI_API_KEY").strip(),
        config_vlm_device=str(merged.get("config_vlm_device") or "").strip().lower(),
        config_vlm_crop_feedback_enabled=_normalize_bool(merged["config_vlm_crop_feedback_enabled"], "config_vlm_crop_feedback_enabled"),
        config_vlm_crop_cache_size=int(merged["config_vlm_crop_cache_size"]),
        config_vlm_dead_after_lost_frames=int(merged["config_vlm_dead_after_lost_frames"]),
        config_vlm_runtime_mode=str(merged["config_vlm_runtime_mode"]).strip().lower(),
        config_vlm_worker_max_queue_size=int(merged["config_vlm_worker_max_queue_size"]),
        config_vlm_worker_batch_size=int(merged["config_vlm_worker_batch_size"]),
        config_vlm_worker_batch_wait_ms=int(merged["config_vlm_worker_batch_wait_ms"]),
        config_vlm_worker_spill_queue_path=str(merged["config_vlm_worker_spill_queue_path"]).strip(),
        config_vlm_spill_max_file_mb=float(merged["config_vlm_spill_max_file_mb"]),
        config_vlm_realtime_throttle_enabled=_normalize_bool(
            merged["config_vlm_realtime_throttle_enabled"],
            "config_vlm_realtime_throttle_enabled",
        ),
        config_scene_awareness_enabled=_normalize_bool(
            merged["config_scene_awareness_enabled"],
            "config_scene_awareness_enabled",
        ),
        config_metadata_output_enabled=_normalize_bool(
            merged["config_metadata_output_enabled"],
            "config_metadata_output_enabled",
        ),
        config_evaluation_output_enabled=_normalize_bool(
            merged["config_evaluation_output_enabled"],
            "config_evaluation_output_enabled",
        ),
    )
