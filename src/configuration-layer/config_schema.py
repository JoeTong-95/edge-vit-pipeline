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
    "config_yolo_imgsz",
    "config_vlm_enabled",
    "config_vlm_backend",
    "config_vlm_model",
    "config_vlm_api_key_env",
    "config_vlm_device",
    "config_vlm_crop_feedback_enabled",
    "config_vlm_crop_cache_size",
    "config_vlm_dead_after_lost_frames",
    "config_vlm_runtime_mode",
    "config_vlm_worker_max_queue_size",
    "config_vlm_worker_batch_size",
    "config_vlm_worker_batch_wait_ms",
    "config_vlm_worker_spill_queue_path",
    "config_vlm_spill_max_file_mb",
    "config_vlm_realtime_throttle_enabled",
    "config_scene_awareness_enabled",
    "config_metadata_output_enabled",
    "config_evaluation_output_enabled",
}

CONFIG_ALLOWED_VLM_RUNTIME_MODES = {"inline", "async", "spill"}
CONFIG_ALLOWED_VLM_BACKENDS = {
    "auto",
    "huggingface_local",
    "smolvlm_256m",
    "qwen_0_8b",
    "gemma_e2b_local",
    "gemini_e2b",
}

CONFIG_ALLOWED_DEVICES = {"cpu", "cuda"}
CONFIG_ALLOWED_INPUT_SOURCES = {"camera", "video"}
CONFIG_REQUIRED_PATH_INPUT_SOURCE = "video"
