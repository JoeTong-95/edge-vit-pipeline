"""
input_layer.py

Layer: input_layer
Purpose: Normalize all frame sources into one shared input package for the
         rest of the pipeline.

Public functions (per pipeline_layers_and_interactions.md):
    initialize_input_layer      — prepare the selected input source using config values.
    read_next_frame             — read the next available frame from the active input node.
    build_input_layer_package   — normalize raw frame data into the shared input package.
    close_input_layer           — release the active input source.

Interacts with:
    configuration_layer  (via config_stub until real layer is ready)
    roi_layer            (downstream consumer)
    scene_awareness_layer (optional downstream consumer)
    vlm_frame_cropper_layer (downstream consumer)

Config parameters used:
    config_input_source      — selects camera_input_node or video_file_node.
    config_input_path        — video file path when source is "video".
    config_frame_resolution  — target (width, height) for output frames.
"""

import time
from typing import Tuple

import cv2
import numpy as np

try:
    from .video_file_node import VideoFileNode
    from .camera_input_node import CameraInputNode
    from .input_layer_package import InputLayerPackage
except ImportError:
    from video_file_node import VideoFileNode
    from camera_input_node import CameraInputNode
    from input_layer_package import InputLayerPackage


class InputLayer:
    """Orchestrates frame ingestion from either a video file or live camera."""

    def __init__(self):
        self._source_type: str = ""
        self._resolution: Tuple[int, int] = (640, 480)
        self._frame_counter: int = 0

        # Nodes
        self._video_node: VideoFileNode | None = None
        self._camera_node: CameraInputNode | None = None

        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Public functions
    # ------------------------------------------------------------------

    def initialize_input_layer(
        self,
        config_input_source: str,
        config_frame_resolution: Tuple[int, int],
        config_input_path: str = "",
        camera_device_index: int = 0,
        use_gstreamer: bool = False,
    ) -> None:
        """Prepare the selected input source using config values.

        Parameters
        ----------
        config_input_source : str
            ``"video"`` or ``"camera"``.
        config_frame_resolution : Tuple[int, int]
            Target (width, height) for every frame emitted.
        config_input_path : str
            Path to the video file (required when source is ``"video"``).
        camera_device_index : int
            Camera index or GStreamer sensor-id (used when source is ``"camera"``).
        use_gstreamer : bool
            If True, use GStreamer pipeline for Jetson CSI camera.
        """
        self._source_type = config_input_source
        self._resolution = tuple(config_frame_resolution)
        self._frame_counter = 0

        if self._source_type == "video":
            self._video_node = VideoFileNode()
            self._video_node.open_video_file(config_input_path)
        elif self._source_type == "camera":
            self._camera_node = CameraInputNode()
            self._camera_node.open_camera_stream(
                device_index=camera_device_index,
                use_gstreamer=use_gstreamer,
            )
        else:
            raise ValueError(
                f"Unknown input source: '{self._source_type}'. "
                f"Expected 'video' or 'camera'."
            )

        self._initialized = True

    def read_next_frame(self) -> np.ndarray | None:
        """Read the next available frame from the active input node.

        Returns
        -------
        np.ndarray or None
            Raw BGR frame, or ``None`` when the source is exhausted / failed.
        """
        self._assert_initialized()

        if self._source_type == "video":
            return self._video_node.read_video_frame()
        else:
            return self._camera_node.read_camera_frame()

    def build_input_layer_package(
        self, raw_frame: np.ndarray
    ) -> InputLayerPackage:
        """Normalize raw frame data into the shared input package.

        Resizes the frame to ``config_frame_resolution`` and wraps it with
        the required metadata fields.

        Parameters
        ----------
        raw_frame : np.ndarray
            BGR uint8 frame straight from the input node.

        Returns
        -------
        InputLayerPackage
        """
        self._assert_initialized()

        # Resize to target resolution (width, height)
        target_w, target_h = self._resolution
        resized = cv2.resize(raw_frame, (target_w, target_h))

        self._frame_counter += 1

        return InputLayerPackage(
            input_layer_frame_id=self._frame_counter,
            input_layer_timestamp=time.time(),
            input_layer_image=resized,
            input_layer_source_type=self._source_type,
            input_layer_resolution=self._resolution,
        )

    def close_input_layer(self) -> None:
        """Release the active input source."""
        if self._video_node is not None:
            self._video_node.close_video_file()
            self._video_node = None
        if self._camera_node is not None:
            self._camera_node.close_camera_stream()
            self._camera_node = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assert_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "input_layer has not been initialized. "
                "Call initialize_input_layer first."
            )

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def frame_count(self) -> int:
        return self._frame_counter
