from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_VLM_UTIL_DIR = Path(__file__).resolve().parent.parent / "util"
if str(_VLM_UTIL_DIR) not in sys.path:
    sys.path.insert(0, str(_VLM_UTIL_DIR))

from vlm_deferred_queue import DeferredVLMTask, append_deferred_task, maybe_rotate_spill_file


class SpillRotationTests(unittest.TestCase):
    def test_maybe_rotate_renames_when_at_or_over_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = Path(tmp) / "q.jsonl"
            q.write_text("x" * 100, encoding="utf-8")
            rotated = maybe_rotate_spill_file(q, max_file_bytes=50)
            self.assertIsNotNone(rotated)
            self.assertTrue(rotated.is_file())
            self.assertFalse(q.exists())

    def test_append_rotates_before_write_when_needed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            q = Path(tmp) / "q.jsonl"
            q.write_text("x" * 200, encoding="utf-8")
            task = DeferredVLMTask(
                track_id="1",
                dispatch_frame_id=1,
                query_type="t",
                prompt_text="p",
                crop_png_base64="",
            )
            append_deferred_task(q, task, max_file_bytes=100)
            self.assertTrue(q.is_file())
            rotated = list(Path(tmp).glob("*.rotated.*"))
            self.assertEqual(len(rotated), 1)
            self.assertIn('{"track_id": "1"', q.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
