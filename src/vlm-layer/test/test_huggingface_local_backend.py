from __future__ import annotations

import sys
import unittest
from pathlib import Path


_BACKENDS_DIR = Path(__file__).resolve().parent.parent / "backends"
if str(_BACKENDS_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKENDS_DIR))

import huggingface_local


class HuggingFaceLocalBackendTests(unittest.TestCase):
    def test_retry_without_device_map_only_for_accelerate_style_errors(self) -> None:
        self.assertTrue(
            huggingface_local._should_retry_without_device_map(
                RuntimeError("Using a `device_map` requires `accelerate`.")
            )
        )
        self.assertTrue(
            huggingface_local._should_retry_without_device_map(
                ImportError("No module named 'accelerate'")
            )
        )
        self.assertFalse(
            huggingface_local._should_retry_without_device_map(
                RuntimeError("CUDA error: CUBLAS_STATUS_EXECUTION_FAILED")
            )
        )

    def test_jetson_cuda_move_failure_is_reframed_cleanly(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "failed during full-model placement"):
            huggingface_local._raise_if_jetson_cuda_move_failure(
                exc=RuntimeError("NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:1131"),
                model_path=Path("/tmp/Qwen3.5-0.8B-local-hydrated"),
                runtime_device="cuda",
                runtime_dtype="torch.float16",
            )

    def test_non_cuda_move_failure_is_left_untouched(self) -> None:
        huggingface_local._raise_if_jetson_cuda_move_failure(
            exc=RuntimeError("NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:1131"),
            model_path=Path("/tmp/Qwen3.5-0.8B-local-hydrated"),
            runtime_device="cpu",
            runtime_dtype="torch.float32",
        )

    def test_jetson_cuda_runtime_failure_is_reframed_cleanly(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Treat this as a real CUDA runtime"):
            huggingface_local._raise_if_jetson_cuda_runtime_failure(
                exc=RuntimeError("CUDA error: CUBLAS_STATUS_EXECUTION_FAILED"),
                runtime_device="cuda",
                model_id="Qwen3.5-0.8B-local-hydrated",
                stage="generate",
            )


if __name__ == "__main__":
    unittest.main()
