from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

EVALUATION_LAYER_DIR = Path(__file__).resolve().parent.parent
if str(EVALUATION_LAYER_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_LAYER_DIR))

from evaluation_output_layer import (
    build_evaluation_output_layer_package,
    collect_evaluation_metrics,
    emit_evaluation_output,
)


class EvaluationOutputLayerTests(unittest.TestCase):
    def test_build_and_emit_sqlite_record(self) -> None:
        metrics = collect_evaluation_metrics(
            input_layer_package={"input_layer_frame_id": "frame-1", "input_layer_timestamp": 1.0},
            yolo_layer_package={"yolo_layer_detections": [{"id": 1}, {"id": 2}]},
            tracking_layer_package=[{"track": 1}],
            timings={"pipeline_total_s": 0.5, "module_latency": {"yolo": 0.1}},
        )
        package = build_evaluation_output_layer_package(metrics)

        self.assertEqual(package["evaluation_output_layer_detection_count"], 2)
        self.assertEqual(package["evaluation_output_layer_track_count"], 1)
        self.assertEqual(package["evaluation_output_layer_frame_id"], "frame-1")

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "evaluation.sqlite"
            emit_evaluation_output(package, output_destination="sqlite", output_path=str(db_path))
            self.assertTrue(db_path.is_file())


if __name__ == "__main__":
    unittest.main()
