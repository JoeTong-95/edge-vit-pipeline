"""
input_layer_package.py

Defines the data contract passed from the input layer to all downstream layers.
Field names follow the naming convention: input_layer_<human_readable_parameter>
as required by pipeline_layers_and_interactions.md.
"""

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class InputLayerPackage:
    """Package produced by the input layer for downstream consumption.

    Fields
    ------
    input_layer_frame_id : int
        Unique identifier for the current frame (monotonically increasing).
    input_layer_timestamp : float
        Capture or ingest time associated with the frame (epoch seconds).
    input_layer_image : np.ndarray
        Raw frame image passed into the pipeline (BGR, uint8).
    input_layer_source_type : str
        Source label — either ``"camera"`` or ``"video"``.
    input_layer_resolution : Tuple[int, int]
        Active (width, height) for the frame after resizing.
    """

    input_layer_frame_id: int
    input_layer_timestamp: float
    input_layer_image: np.ndarray
    input_layer_source_type: str
    input_layer_resolution: Tuple[int, int]

    # Prevent numpy arrays from being compared element-wise in __eq__
    def __eq__(self, other):
        if not isinstance(other, InputLayerPackage):
            return NotImplemented
        return (
            self.input_layer_frame_id == other.input_layer_frame_id
            and self.input_layer_timestamp == other.input_layer_timestamp
            and self.input_layer_source_type == other.input_layer_source_type
            and self.input_layer_resolution == other.input_layer_resolution
            and np.array_equal(self.input_layer_image, other.input_layer_image)
        )

    def __repr__(self):
        h, w = self.input_layer_image.shape[:2]
        return (
            f"InputLayerPackage("
            f"frame_id={self.input_layer_frame_id}, "
            f"timestamp={self.input_layer_timestamp:.3f}, "
            f"source={self.input_layer_source_type}, "
            f"resolution={self.input_layer_resolution}, "
            f"image_shape=({h}, {w}))"
        )
