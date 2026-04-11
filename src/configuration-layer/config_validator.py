from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from config_schema import (
    CONFIG_ALLOWED_DEVICES,
    CONFIG_ALLOWED_INPUT_SOURCES,
    CONFIG_ALLOWED_KEYS,
    CONFIG_REQUIRED_PATH_INPUT_SOURCE,
)
from config_types import ConfigurationLayerConfig

_CONFIG_LAYER_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CONFIG_LAYER_DIR.parent.parent


def _to_mapping(config: ConfigurationLayerConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, ConfigurationLayerConfig):
        return config.to_dict()
    return dict(config)


def validate_config_values(config: ConfigurationLayerConfig | Mapping[str, Any]) -> None:
    config_values = _to_mapping(config)

    unknown_keys = sorted(set(config_values) - CONFIG_ALLOWED_KEYS)
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {unknown_keys}")

    missing_keys = sorted(CONFIG_ALLOWED_KEYS - set(config_values))
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

    if config_values["config_device"] not in CONFIG_ALLOWED_DEVICES:
        raise ValueError(
            f"config_device must be one of {sorted(CONFIG_ALLOWED_DEVICES)}."
        )

    if config_values["config_input_source"] not in CONFIG_ALLOWED_INPUT_SOURCES:
        raise ValueError(
            f"config_input_source must be one of {sorted(CONFIG_ALLOWED_INPUT_SOURCES)}."
        )

    if (
        config_values["config_input_source"] == CONFIG_REQUIRED_PATH_INPUT_SOURCE
        and not config_values["config_input_path"]
    ):
        raise ValueError(
            "config_input_path is required when config_input_source is 'video'."
        )

    if config_values["config_input_source"] == CONFIG_REQUIRED_PATH_INPUT_SOURCE:
        input_path = Path(config_values["config_input_path"])
        candidate_paths = [input_path]
        if not input_path.is_absolute():
            candidate_paths.append(_REPO_ROOT / input_path)

        if not any(path.exists() for path in candidate_paths):
            raise FileNotFoundError(
                f"config_input_path does not exist: {config_values['config_input_path']}"
            )

    if config_values["config_input_source"] == "camera" and config_values["config_input_path"]:
        raise ValueError(
            "config_input_path must be empty when config_input_source is 'camera'."
        )

    width, height = config_values["config_frame_resolution"]
    if width <= 0 or height <= 0:
        raise ValueError("config_frame_resolution must contain positive integers.")

    if config_values["config_roi_vehicle_count_threshold"] <= 0:
        raise ValueError("config_roi_vehicle_count_threshold must be greater than 0.")

    confidence = config_values["config_yolo_confidence_threshold"]
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("config_yolo_confidence_threshold must be between 0.0 and 1.0.")

    if config_values["config_vlm_enabled"] and not config_values["config_vlm_model"]:
        raise ValueError("config_vlm_model is required when config_vlm_enabled is true.")

    if not isinstance(config_values["config_vlm_crop_feedback_enabled"], bool):
        raise ValueError("config_vlm_crop_feedback_enabled must be a bool.")

    if int(config_values["config_vlm_crop_cache_size"]) <= 0:
        raise ValueError("config_vlm_crop_cache_size must be greater than 0.")

    if int(config_values["config_vlm_dead_after_lost_frames"]) <= 0:
        raise ValueError("config_vlm_dead_after_lost_frames must be greater than 0.")
