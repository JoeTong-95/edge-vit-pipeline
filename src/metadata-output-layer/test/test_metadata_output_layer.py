from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

METADATA_LAYER_DIR = Path(__file__).resolve().parent.parent
if str(METADATA_LAYER_DIR) not in sys.path:
    sys.path.insert(0, str(METADATA_LAYER_DIR))

from metadata_output_layer import (
    build_metadata_output_layer_package,
    emit_metadata_output,
    serialize_metadata_output,
)


class MetadataOutputLayerTests(unittest.TestCase):
    def test_build_serialize_and_emit_file(self) -> None:
        vehicle_state_layer_package = {
            "vehicle_state_layer_track_id": [2, 1],
            "vehicle_state_layer_vehicle_class": ["truck", "car"],
            "vehicle_state_layer_semantic_tags": [["white"], ["sedan"]],
            "vehicle_state_layer_vlm_called": [True, False],
        }
        vlm_layer_package = {
            "vlm_layer_track_id": [2],
            "vlm_layer_label": ["box_truck"],
            "vlm_layer_attributes": [{"color": "white"}],
        }

        package = build_metadata_output_layer_package(
            vehicle_state_layer_package=vehicle_state_layer_package,
            vlm_layer_package=vlm_layer_package,
        )
        self.assertEqual(package["metadata_output_layer_object_ids"], ["1", "2"])

        payload = serialize_metadata_output(package, output_format="json")
        decoded = json.loads(payload)
        self.assertIn("metadata_output_layer_counts", decoded)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "metadata.jsonl"
            emit_metadata_output(payload, output_destination="file", output_path=str(out_path))
            self.assertTrue(out_path.is_file())


if __name__ == "__main__":
    unittest.main()
