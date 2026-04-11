from __future__ import annotations

DEFAULT_CONFIG_VALUES = {
    "config_device": "cpu",
    "config_input_source": "video",
    "config_input_path": "data/sample.mp4",
    "config_frame_resolution": (640, 480),
    "config_roi_enabled": False,
    "config_roi_vehicle_count_threshold": 5,
    "config_yolo_model": "yolov8n.pt",
    "config_yolo_confidence_threshold": 0.25,
    "config_vlm_enabled": False,
    "config_vlm_model": "",
    "config_vlm_crop_feedback_enabled": True,
    "config_vlm_crop_cache_size": 5,
    "config_vlm_dead_after_lost_frames": 3,
    "config_scene_awareness_enabled": False,
    "config_metadata_output_enabled": True,
    "config_evaluation_output_enabled": False,
}
