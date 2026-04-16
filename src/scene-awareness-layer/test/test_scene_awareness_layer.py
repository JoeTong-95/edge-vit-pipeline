from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCENE_LAYER_DIR = Path(__file__).resolve().parent.parent
if str(SCENE_LAYER_DIR) not in sys.path:
    sys.path.insert(0, str(SCENE_LAYER_DIR))

from scene_awareness_layer import initialize_scene_awareness_layer, run_scene_awareness_inference


class SceneAwarenessLayerTests(unittest.TestCase):
    def test_enabled_runtime_builds_scene_package(self) -> None:
        runtime = initialize_scene_awareness_layer(True, "cpu")
        package = run_scene_awareness_inference(
            runtime,
            {
                "input_layer_frame_id": "frame-1",
                "input_layer_timestamp": 1.0,
                "input_layer_image": [[[0, 0, 0], [255, 255, 255]]],
            },
        )

        self.assertIsNotNone(package)
        assert package is not None
        self.assertEqual(package["scene_awareness_layer_frame_id"], "frame-1")
        self.assertTrue(package["scene_awareness_layer_label"])

    def test_disabled_runtime_returns_none(self) -> None:
        runtime = initialize_scene_awareness_layer(False, "cpu")
        package = run_scene_awareness_inference(runtime, {"input_layer_image": [[[0, 0, 0]]]})
        self.assertIsNone(package)


if __name__ == "__main__":
    unittest.main()
