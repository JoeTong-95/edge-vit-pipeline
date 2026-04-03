"""
video_file_node.py

Node: video_file_node
Purpose: Read frames from a recorded video source.

Internal functions (per pipeline_layers_and_interactions.md):
    open_video_file  — open the configured video file.
    read_video_frame — return the next frame from the file.
    close_video_file — close the video file handle.

Interacts with: input_layer
"""

import cv2
import numpy as np


class VideoFileNode:
    """Read frames sequentially from a video file using OpenCV."""

    def __init__(self):
        self._cap: cv2.VideoCapture | None = None
        self._path: str = ""

    # ------------------------------------------------------------------
    # Internal functions
    # ------------------------------------------------------------------

    def open_video_file(self, path: str) -> None:
        """Open the configured video file.

        Parameters
        ----------
        path : str
            Filesystem path to the video file.

        Raises
        ------
        FileNotFoundError
            If *path* does not point to a readable file.
        RuntimeError
            If OpenCV cannot open the file.
        """
        self._path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video file: {path}")

    def read_video_frame(self) -> np.ndarray | None:
        """Return the next frame from the file.

        Returns
        -------
        np.ndarray or None
            BGR uint8 frame, or ``None`` when the video is exhausted.
        """
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def close_video_file(self) -> None:
        """Close the video file handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Helpers (not part of the spec, but useful)
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def total_frames(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
