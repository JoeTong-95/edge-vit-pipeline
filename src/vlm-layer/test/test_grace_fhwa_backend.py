from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


BACKENDS_DIR = Path(__file__).resolve().parent.parent / "backends"
if str(BACKENDS_DIR) not in sys.path:
    sys.path.insert(0, str(BACKENDS_DIR))

import grace_fhwa
import registry


class GraceFHWABackendTests(unittest.TestCase):
    def test_registry_resolves_grace_backend(self) -> None:
        self.assertEqual(
            registry.resolve_vlm_backend_name("grace_fhwa", "src/vlm-layer/grace_integration"),
            "grace_fhwa",
        )
        self.assertEqual(
            registry.resolve_vlm_backend_runtime_kind("grace_fhwa", "src/vlm-layer/grace_integration"),
            "grace_fhwa",
        )
        self.assertEqual(
            registry.resolve_vlm_backend_name("auto", "src/vlm-layer/grace_integration"),
            "grace_fhwa",
        )

    def test_load_target_vehicle_types_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target_path = Path(tmp) / "target_vehicle_types.yaml"
            target_path.write_text(
                "target_vehicle_types:\n  - semi_tractor\n  - dump_truck\n",
                encoding="utf-8",
            )
            self.assertEqual(
                grace_fhwa.load_target_vehicle_types(target_path),
                {"semi_tractor", "dump_truck"},
            )

    def test_convert_grace_result_to_vlm_json(self) -> None:
        raw = grace_fhwa.convert_grace_result_to_vlm_json(
            {
                "fhwa_class": "FHWA-9",
                "fhwa_index": 7,
                "fhwa_confidence": 0.94444,
                "vehicle_type": "semi_tractor",
                "vehicle_type_index": 14,
                "axle_count": 5.0,
                "trailer_count": "1",
            },
            target_vehicle_types={"semi_tractor"},
        )
        payload = json.loads(raw)
        self.assertTrue(payload["is_target_vehicle"])
        self.assertEqual(payload["axle_count"], 5.0)
        self.assertEqual(payload["fhwa_class"], "FHWA-9")
        self.assertEqual(payload["ack_status"], "accepted")
        self.assertEqual(payload["retry_reasons"], [])
        self.assertNotIn("estimated_weight_kg", payload)
        self.assertNotIn("wheel_count", payload)


if __name__ == "__main__":
    unittest.main()
