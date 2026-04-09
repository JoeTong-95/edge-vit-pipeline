from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from config_loader import load_raw_config
from config_normalizer import normalize_config
from config_types import ConfigurationLayerConfig
from config_validator import validate_config_values

__all__ = ["load_config", "validate_config", "get_config_value"]


def load_config(
    config_source: str | Path | Mapping[str, Any] | None = None,
) -> ConfigurationLayerConfig:
    raw_config = load_raw_config(config_source)
    normalized_config = normalize_config(raw_config)
    return normalized_config


def validate_config(config: ConfigurationLayerConfig | Mapping[str, Any]) -> None:
    validate_config_values(config)


def get_config_value(
    config: ConfigurationLayerConfig | Mapping[str, Any],
    config_key: str,
) -> Any:
    if isinstance(config, ConfigurationLayerConfig):
        config_values = config.to_dict()
    else:
        config_values = dict(config)

    if config_key not in config_values:
        raise KeyError(f"Unknown config key: {config_key}")

    return config_values[config_key]
