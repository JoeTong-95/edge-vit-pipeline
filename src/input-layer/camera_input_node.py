"""
camera_input_node.py

Node: camera_input_node
Purpose: Read frames from a live camera source.

Internal functions (per pipeline_layers_and_interactions.md):
    open_camera_stream  — open the configured camera device.
    read_camera_frame   — return the next camera frame.
    close_camera_stream — close the camera device.

Interacts with: input_layer

Notes on Jetson + RPi Camera Module 3 (CSI):
    On the Jetson Orin Nano the camera is accessed via GStreamer through
    OpenCV.  The GStreamer pipeline string is constructed automatically
    when ``use_gstreamer=True`` is passed to ``open_camera_stream``.
    On a dev laptop a regular V4L2 / DirectShow index (typically 0) is
    used instead.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# GStreamer pipeline template for Jetson CSI cameras
# ---------------------------------------------------------------------------

_GSTREAMER_PIPELINE_TEMPLATE = (
    "nvarguscamerasrc sensor-id={sensor_id} ! "
    "video/x-raw(memory:NVMM), width={width}, height={height}, "
    "format=NV12, framerate={fps}/1 ! "
    "nvvidconv flip-method={flip_method} ! "
    "video/x-raw, width={width}, height={height}, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)


class CameraInputNode:
    """Read frames from a live camera device using OpenCV."""

    def __init__(self):
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Internal functions
    # ------------------------------------------------------------------

    def open_camera_stream(
        self,
        device_index: int = 0,
        *,
        use_gstreamer: bool = False,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        flip_method: int = 0,
    ) -> None:
        """Open the configured camera device.

        Parameters
        ----------
        device_index : int
            Camera index (V4L2 / DirectShow) or ``sensor-id`` for GStreamer.
        use_gstreamer : bool
            If ``True``, build a GStreamer pipeline for Jetson CSI cameras.
        width, height, fps, flip_method
            Only used when *use_gstreamer* is ``True``.

        Raises
        ------
        RuntimeError
            If OpenCV cannot open the camera.
        """
        if use_gstreamer:
            pipeline = _GSTREAMER_PIPELINE_TEMPLATE.format(
                sensor_id=device_index,
                width=width,
                height=height,
                fps=fps,
                flip_method=flip_method,
            )
            self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self._cap = cv2.VideoCapture(device_index)

        if not self._cap.isOpened():
            source = f"GStreamer sensor {device_index}" if use_gstreamer else f"camera index {device_index}"
            raise RuntimeError(f"OpenCV failed to open {source}")

    def read_camera_frame(self) -> np.ndarray | None:
        """Return the next camera frame.

        Returns
        -------
        np.ndarray or None
            BGR uint8 frame, or ``None`` if the read fails.
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def close_camera_stream(self) -> None:
        """Close the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
