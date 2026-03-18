from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_raw_config(config_source: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if config_source is None:
        return {}

    if isinstance(config_source, Mapping):
        return dict(config_source)

    config_path = Path(config_source)
    if not config_path.exists():
        raise FileNotFoundError(f"Config source not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix == ".json":
        return json.loads(config_path.read_text(encoding="utf-8"))

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "YAML config loading requires PyYAML, or use a JSON config file instead."
            ) from exc

        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return {} if loaded is None else dict(loaded)

    raise ValueError(
        f"Unsupported config file extension '{suffix}'. Use .json, .yaml, or .yml."
    )
