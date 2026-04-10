from __future__ import annotations

CONFIG_ALLOWED_KEYS = {
    "config_device",
    "config_input_source",
    "config_input_path",
    "config_frame_resolution",
    "config_roi_enabled",
    "config_roi_vehicle_count_threshold",
    "config_yolo_model",
    "config_yolo_confidence_threshold",
    "config_vlm_enabled",
    "config_vlm_model",
    "config_vlm_crop_feedback_enabled",
    "config_vlm_crop_cache_size",
    "config_scene_awareness_enabled",
    "config_metadata_output_enabled",
    "config_evaluation_output_enabled",
}

CONFIG_ALLOWED_DEVICES = {"cpu", "cuda"}
CONFIG_ALLOWED_INPUT_SOURCES = {"camera", "video"}
CONFIG_REQUIRED_PATH_INPUT_SOURCE = "video"
