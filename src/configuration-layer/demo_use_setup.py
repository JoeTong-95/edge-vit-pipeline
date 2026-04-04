"""Example usage of the configuration layer.

Other developers may refer to this file as a simple example of how to use
`load_config`, `validate_config`, and `get_config_value` from `config_node`.
This script demonstrates the intended pattern where configuration is loaded
first and then passed into a downstream layer's public function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config_node import get_config_value, load_config, validate_config

CONFIG_LAYER_DIR = Path(__file__).resolve().parent


def initialize_input_layer(
    config_input_source: str,
    config_input_path: str | None,
    config_frame_resolution: tuple[int, int],
) -> dict[str, Any]:
    active_input_node = (
        "camera_input_node" if config_input_source == "camera" else "video_file_node"
    )

    return {
        "input_layer_status": "initialized",
        "input_layer_active_input_node": active_input_node,
        "input_layer_selected_source": config_input_source,
        "input_layer_selected_path": config_input_path,
        "input_layer_selected_resolution": config_frame_resolution,
    }


def main() -> None:
    config_path = CONFIG_LAYER_DIR / "config.yaml"
    config = load_config(config_path)
    validate_config(config)

    input_layer_result = initialize_input_layer(
        config_input_source=get_config_value(config, "config_input_source"),
        config_input_path=get_config_value(config, "config_input_path"),
        config_frame_resolution=get_config_value(config, "config_frame_resolution"),
    )

    print("configuration_layer -> input_layer smoke test")
    print(f"config_input_source={input_layer_result['input_layer_selected_source']}")
    print(f"config_input_path={input_layer_result['input_layer_selected_path']}")
    print(
        "config_frame_resolution="
        f"{input_layer_result['input_layer_selected_resolution']}"
    )
    print(
        "input_layer initialized with "
        f"{input_layer_result['input_layer_active_input_node']}"
    )
    print(f"input_layer_status={input_layer_result['input_layer_status']}")


if __name__ == "__main__":
    main()
