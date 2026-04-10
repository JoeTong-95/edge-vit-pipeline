from __future__ import annotations

import sys
import unittest
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config_node import get_config_value, load_config, validate_config


class ConfigurationLayerTests(unittest.TestCase):
    def test_load_config_applies_defaults(self) -> None:
        config = load_config({"config_input_path": "data/sample.mp4"})
        self.assertEqual(config.config_device, "cpu")
        self.assertEqual(config.config_input_source, "video")
        self.assertEqual(config.config_frame_resolution, (640, 480))
        self.assertTrue(config.config_vlm_crop_feedback_enabled)

    def test_validate_config_accepts_valid_config(self) -> None:
        config = load_config(
            {
                "config_device": "cuda",
                "config_input_source": "camera",
                "config_input_path": None,
                "config_vlm_enabled": True,
                "config_vlm_model": "example-vlm",
                "config_vlm_crop_feedback_enabled": False,
            }
        )
        validate_config(config)

    def test_validate_config_rejects_missing_video_path(self) -> None:
        config = load_config(
            {
                "config_input_source": "video",
                "config_input_path": None,
            }
        )
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_get_config_value_returns_exact_key(self) -> None:
        config = load_config({"config_input_path": "data/sample.mp4"})
        self.assertEqual(get_config_value(config, "config_input_source"), "video")


if __name__ == "__main__":
    unittest.main()
