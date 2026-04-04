"""
input-layer package

Public API:
    InputLayer          — main orchestrator
    InputLayerPackage   — data contract for downstream layers
    VideoFileNode       — video file input node
    CameraInputNode     — camera input node
"""

try:
    from .input_layer import InputLayer
    from .input_layer_package import InputLayerPackage
    from .video_file_node import VideoFileNode
    from .camera_input_node import CameraInputNode
except ImportError:
    from input_layer import InputLayer
    from input_layer_package import InputLayerPackage
    from video_file_node import VideoFileNode
    from camera_input_node import CameraInputNode

__all__ = [
    "InputLayer",
    "InputLayerPackage",
    "VideoFileNode",
    "CameraInputNode",
]
