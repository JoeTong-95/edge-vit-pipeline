from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class ConfigurationLayerConfig:
    config_device: str
    config_input_source: str
    config_input_path: str | None
    config_frame_resolution: Tuple[int, int]
    config_roi_enabled: bool
    config_roi_vehicle_count_threshold: int
    config_yolo_model: str
    config_yolo_confidence_threshold: float
    config_vlm_enabled: bool
    config_vlm_model: str
    config_vlm_crop_feedback_enabled: bool
    config_vlm_crop_cache_size: int
    config_scene_awareness_enabled: bool
    config_metadata_output_enabled: bool
    config_evaluation_output_enabled: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
