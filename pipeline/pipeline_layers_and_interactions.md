# Edge VLM Pipeline Layers and Interactions

## Purpose

This document defines the semantic layers, nodes, functions, packages, and interactions in the current Edge VLM pipeline.

It does not define internal implementation details such as exact classes, helper utilities, or file layout inside each node.

The goal is to make clear:

- what each layer is responsible for
- what each node is responsible for
- what function names each layer should expose
- what each function should do
- what packages and parameters move between layers
- which layer owns each decision or metadata field

## Core Principles

- The focus is on transactions between layers, not internal implementation.
- `tracking` decides whether an object is new or existing.
- `vehicle_state_manager` stores persistent metadata and does not decide whether an object is new.
- `vlm` is an optional enrichment path, not the primary perception path.
- Package and variable names should follow `node_name_human_readable_parameter`.
- Public function names should be action-oriented and stable.
- Naming should use lowercase `snake_case` consistently.

## Terminology

### Layer

A layer is a semantic module that interacts with one or more other modules in the pipeline.

A layer may:

- receive packages from upstream modules
- send packages to downstream modules
- depend on config settings
- coordinate with multiple adjacent modules

Examples:

- `input_layer`
- `roi_layer`
- `vlm_layer`
- `metadata_output_layer`

### Node

A node is a sub-part inside a layer that has a narrower purpose and usually interacts only within that layer or directly with its parent layer.

A node may:

- perform one focused task
- serve one specific source or sink
- stay internal to its parent layer

Examples:

- `video_file_node` inside `input_layer`
- `camera_input_node` inside `input_layer`

### Package

A package is the named data contract exchanged between layers.

A package should:

- have a stable name
- contain only the fields needed by downstream layers
- use node-oriented parameter names

## Suggested workload break-down
Person A: configuration_layer, input_layer, roi_layer
Why:
These are the front-end ingestion and preprocessing path.
They are tightly coupled through frame sourcing, frame normalization, and ROI application.
One person owning this avoids mismatches in frame format and ROI package structure.

Person B: yolo_layer, tracking_layer
Why:
These are the core perception path.
Detection and tracking need very tight agreement on package format and timing.
Keeping them together reduces integration pain.

Person C: vlm_frame_cropper_layer, vlm_layer, scene_awareness_layer
Why:
These are the semantic/AI-heavy paths.
They share model-loading concerns, inference concerns, and image-prep concerns.
This person can own all optional semantic reasoning logic.

Person D: vehicle_state_layer, metadata_output_layer, evaluation_output_layer
Why:
These are the coordination and output layers.
They define what the system remembers, emits, and measures.
This is a strong integration role because these layers sit at the boundary between all other contributors.

## Layer Overview


## VLM Semantic Gate

The current VLM contract is intentionally narrow:

- first decide whether the crop is one of the currently active YOLO labels, such as `truck` or `bus`
- if not, return `is_truck=false` and acknowledge the track with reason `no`
- if yes, return a small JSON payload with only:
  - `wheel_count`
  - `ack_status`
  - `retry_reasons`

Important notes:

- the allowed label set comes from the currently active detector-label filter, not from a separate semantic vocabulary
- `retry_reasons` is the structured explanation for a bad crop
- the currently allowed retry reasons are:
  - `occluded`
  - `bad_angle`
- free-text image-quality notes are not part of the current VLM contract


### 1. Configuration Layer

#### config_node

Purpose:

- Select runtime mode and enabled layers.
- Define the minimum set of behavior-changing parameters.

Public functions:

- `load_config`: read the config source and return normalized config values.
- `validate_config`: check required parameters, allowed values, and layer compatibility.
- `get_config_value`: return one named config value for downstream layer use.

Typical controls:

- `config_device`: compute target, such as `cpu` or `cuda`.
- `config_input_source`: frame source type, such as `camera` or `video`.
- `config_input_path`: file path used when `config_input_source` is `video`.
- `config_frame_resolution`: target frame resolution used by the pipeline.
- `config_roi_enabled`: enable or disable ROI-based cropping.
- `config_roi_vehicle_count_threshold`: number of unique-ish startup vehicle detections required before ROI is locked.
- `config_yolo_model`: YOLO model variant used for vehicle detection.
- `config_yolo_confidence_threshold`: minimum detection confidence accepted from YOLO.
- `config_vlm_enabled`: enable or disable the object-level VLM enrichment path.
- `config_vlm_model`: VLM model used for semantic inference.
- `config_vlm_crop_feedback_enabled`: controls whether VLM may ask the cropper for a fresh candidate round after the first dispatch.
- `config_vlm_crop_cache_size`: number of candidate crops cached per track before the cropper-side selector dispatches one crop to VLM.
- `config_vlm_dead_after_lost_frames`: number of consecutive `lost` updates after which the cropper-side VLM flow marks a track terminal `dead`.
- `config_scene_awareness_enabled`: enable or disable the optional full-frame scene-awareness path.
- `config_metadata_output_enabled`: enable or disable structured metadata output.
- `config_evaluation_output_enabled`: enable or disable evaluation and telemetry output.

Primary interactions:

- Supply startup and runtime settings to all active layers.

### 2. Input Layer

Purpose:

- Normalize all frame sources into one shared input package for the rest of the pipeline.

Interacts with:

- `configuration_layer`
- `roi_layer`
- optional `scene_awareness_layer`
- `vlm_frame_cropper_layer`

Public functions:

- `initialize_input_layer`: prepare the selected input source using config values.
- `read_next_frame`: read the next available frame from the active input node.
- `build_input_layer_package`: normalize raw frame data into the shared input package.
- `close_input_layer`: release the active input source.

Suggested external packages:

- `opencv-python`: common choice for camera capture, video decoding, and frame resizing.
- `ffmpeg-python`: useful when video ingest or format handling needs tighter control than OpenCV alone.
- `gstreamer`: useful on Jetson-class devices when hardware-accelerated camera or video pipelines are needed.

Config parameters used:

- `config_input_source`: selects whether `camera_input_node` or `video_file_node` is active.
- `config_input_path`: provides the video file path when `config_input_source` is `video`.
- `config_frame_resolution`: sets the frame size produced by the input layer.

#### input_layer_package

Produces:

- `input_layer_package`

Package fields:

- `input_layer_frame_id`: unique identifier for the current frame.
- `input_layer_timestamp`: capture or ingest time associated with the frame.
- `input_layer_image`: raw frame image passed into the pipeline.
- `input_layer_source_type`: source label, such as `camera` or `video`.
- `input_layer_resolution`: active width and height for the frame.

#### camera_input_node

Purpose:

- Read frames from a live camera source.

Internal functions:

- `open_camera_stream`: open the configured camera device.
- `read_camera_frame`: return the next camera frame.
- `close_camera_stream`: close the camera device.

Interacts with:

- `input_layer`

#### video_file_node

Purpose:

- Read frames from a recorded video source.

Internal functions:

- `open_video_file`: open the configured video file.
- `read_video_frame`: return the next frame from the file.
- `close_video_file`: close the video file handle.

Interacts with:

- `input_layer`

### 3. ROI Layer

Purpose:

- Restrict processing to the relevant scene region before detection.

Interacts with:

- `configuration_layer`
- `input_layer`
- `yolo_layer`

Public functions:

- `initialize_roi_layer`: prepare ROI state from config values.
- `update_roi_state`: update ROI discovery state during startup.
- `apply_roi_to_frame`: apply the active ROI to the input frame.
- `build_roi_layer_package`: create the ROI package for downstream detection.

Suggested external packages:

- `opencv-python`: practical choice for frame cropping, coordinate transforms, and ROI masking.
- `numpy`: useful for bounding-box aggregation and ROI-bound calculations.

Config parameters used:

- `config_roi_enabled`: controls whether ROI cropping is active.
- `config_roi_vehicle_count_threshold`: controls when startup ROI discovery is allowed to lock.

#### roi_layer_package

Produces:

- `roi_layer_package`

Package fields:

- `roi_layer_frame_id`: frame identifier carried forward from input.
- `roi_layer_timestamp`: timestamp carried forward from input.
- `roi_layer_image`: ROI-limited frame image used by downstream detection.
- `roi_layer_bounds`: active ROI bounds applied to the frame.
- `roi_layer_enabled`: whether ROI cropping was active for this frame.
- `roi_layer_locked`: whether startup ROI discovery has already completed.

#### roi_cropper_node

Purpose:

- Apply the active ROI to incoming frames.

Internal functions:

- `crop_frame_to_roi`: crop the frame using the active ROI bounds.
- `pass_through_full_frame`: return the full frame when ROI is disabled or not yet locked.

Interacts with:

- `roi_layer`

#### roi_discovery_node

Purpose:

- Establish the ROI during startup using early-frame vehicle detections.

Internal functions:

- `collect_roi_candidate_boxes`: store candidate vehicle detections during startup.
- `compute_roi_bounds`: compute ROI bounds from collected detections.
- `lock_roi_bounds`: mark ROI discovery as complete and freeze the active ROI.

Parameters used by this node:

- `config_roi_vehicle_count_threshold`: minimum number of unique-ish startup vehicle detections needed before ROI lock.

Important notes:

- Tracking is not required for ROI discovery.
- ROI discovery is startup-focused, while ROI cropping is runtime-focused.

### 4. YOLO Layer

Purpose:

- Detect vehicles in the active frame.

Interacts with:

- `configuration_layer`
- `roi_layer`
- `tracking_layer`
- `roi_layer` during startup ROI discovery support

Public functions:

- `initialize_yolo_layer`: load the configured YOLO model.
- `run_yolo_detection`: run detection on the active frame.
- `filter_yolo_detections`: remove detections below the configured confidence threshold.
- `build_yolo_layer_package`: create the detection package for downstream tracking.

Suggested external packages:

- `ultralytics`: strong default choice for YOLO inference and model management.
- `onnxruntime`: useful if YOLO models are exported to ONNX for portable inference.
- `tensorrt`: useful on Jetson-class hardware when optimizing YOLO inference for deployment.

Config parameters used:

- `config_yolo_model`: selects the detector model loaded by the layer.
- `config_yolo_confidence_threshold`: sets the minimum confidence accepted in `filter_yolo_detections`.
- `config_device`: selects the execution target for YOLO inference.

Important note:

- the bundled YOLO weights may know many labels, but the current repo only forwards classes listed in `src/yolo-layer/class_map.py`
- `src/yolo-layer/TAG_FILTER_BEHAVIOR.md` is the practical reference for what the current filter keeps versus discards
- editing `class_map.py` changes what reaches downstream tracking, cropper, and VLM helper visualizers

#### yolo_layer_package

Produces:

- `yolo_layer_package`

Package fields:

- `yolo_layer_frame_id`: frame identifier associated with the detection set.
- `yolo_layer_detections`: collection of detections generated for the frame.

Suggested contents of each detection:

- `yolo_detection_bbox`: object bounding box in frame coordinates.
- `yolo_detection_class`: detector class label for the object.
- `yolo_detection_confidence`: confidence score for the detection.

#### yolo_detector_node

Purpose:

- Run the selected YOLO model on the active frame.

Internal functions:

- `infer_yolo_detections`: perform model inference on the input image.
- `normalize_yolo_detections`: convert raw model output into the standard detection format.

Interacts with:

- `yolo_layer`

### 5. Tracking Layer

Purpose:

- Maintain object identity across frames and decide object lifecycle state.

Interacts with:

- `yolo_layer`
- `vehicle_state_layer`
- `vlm_frame_cropper_layer`

Public functions:

- `initialize_tracking_layer`: prepare the tracking algorithm state.
- `update_tracks`: associate current detections with existing tracks.
- `assign_tracking_status`: label each tracked object as `new`, `active`, or `lost`.
- `build_tracking_layer_package`: create the tracking package for downstream layers.

Suggested external packages:

- `bytetrack`: strong default candidate for multi-object vehicle tracking.
- `boxmot`: useful wrapper project that includes trackers such as ByteTrack and BoT-SORT.
- `norfair`: lighter-weight tracking option if a simpler tracker abstraction is preferred.

Config parameters used:

- No direct v1 config parameters are required by this layer beyond shared runtime setup.

#### tracking_layer_package

Produces:

- `tracking_layer_package`

Package fields:

- `tracking_layer_frame_id`: frame identifier associated with the tracking update.
- `tracking_layer_track_id`: persistent tracker identifier for the object.
- `tracking_layer_bbox`: latest tracked bounding box.
- `tracking_layer_detector_class`: detector class linked to the tracked object.
- `tracking_layer_confidence`: confidence associated with the current tracking update.
- `tracking_layer_status`: lifecycle state assigned by the tracker.

Allowed values for `tracking_layer_status`:

- `new`: object first seen by the tracker.
- `active`: object is currently tracked and continuing.
- `lost`: object is no longer matched in the current frame window.

Ownership:

- `tracking_layer` owns the decision about whether an object is `new`, `active`, or `lost`.

#### tracking_algorithm_node

Purpose:

- Associate detections across frames and maintain persistent track identities.

Internal functions:

- `match_detections_to_tracks`: link current detections to existing track state.
- `create_new_tracks`: create new track identities for unmatched detections.
- `update_lost_tracks`: increment loss state for unmatched tracked objects.

Interacts with:

- `tracking_layer`

### 6. Vehicle State Layer

Purpose:

- Store persistent per-vehicle metadata across frames.

Interacts with:

- `tracking_layer`
- `vlm_layer`
- `metadata_output_layer`

Public functions:

- `initialize_vehicle_state_layer`: prepare empty persistent vehicle state.
- `update_vehicle_state_from_tracking`: update object state using tracking data.
- `update_vehicle_state_from_vlm`: update stored semantic fields using VLM results.
- `update_vehicle_state_from_vlm_ack`: persist VLM acknowledgement decisions such as retry or finalize.
- `get_vehicle_state_record`: return the current state record for one track.
- `build_vehicle_state_layer_package`: create the vehicle state package for downstream output.

Config parameters used:

- No direct v1 config parameters are required by this layer beyond shared runtime setup.

#### vehicle_state_layer_package

Produces:

- `vehicle_state_layer_package`

Package fields:

- `vehicle_state_layer_track_id`: persistent object identifier used as the state key.
- `vehicle_state_layer_first_seen_frame`: first frame where the tracked object appeared.
- `vehicle_state_layer_last_seen_frame`: most recent frame where the tracked object was updated.
- `vehicle_state_layer_lost_frame_count`: number of consecutive frames where the object has been missing.
- `vehicle_state_layer_vehicle_class`: current stored class for the object.
- `vehicle_state_layer_truck_type`: legacy semantic subtype slot from earlier VLM iterations; currently downstream code may still store the accepted label here for compatibility, but the active VLM prompt no longer asks for subtype classification.
- `vehicle_state_layer_semantic_tags`: stored semantic labels or attributes.
- `vehicle_state_layer_vlm_called`: whether semantic enrichment has already been requested or recorded.
- `vehicle_state_layer_vlm_ack_status`: latest VLM acknowledgement state for the track.
- `vehicle_state_layer_vlm_retry_requested`: whether VLM has explicitly asked for a better crop.
- `vehicle_state_layer_vlm_final_candidate_sent`: whether the cropper already sent the final best-available crop after the object left scope.
- `vehicle_state_layer_terminal_status`: persistent terminal state for the semantic path, such as `tracking`, `no`, `done`, or `dead`.

Ownership:

- `vehicle_state_layer` owns persistent metadata.
- `vehicle_state_layer` does not own the new-object decision.

#### vehicle_state_manager_node

Purpose:

- Update and store the persistent metadata record for each tracked object.

Internal functions:

- `create_vehicle_state_record`: create the initial state record for a newly tracked object.
- `merge_tracking_into_vehicle_state`: apply tracking updates to the stored state.
- `merge_vlm_into_vehicle_state`: apply semantic enrichment to the stored state.
- `merge_vlm_ack_into_vehicle_state`: apply retry or finalize acknowledgements to the stored state.
- `prune_vehicle_state_records`: remove or archive stale state records when needed.

Interacts with:

- `vehicle_state_layer`

### 7. VLM Frame Cropper Layer

Purpose:

- Prepare object-level image crops for semantic reasoning.

Interacts with:

- `tracking_layer`
- `input_layer`
- `vlm_layer`

Public functions:

- `build_vlm_frame_cropper_request_package`: create the request package for object cropping.
- `extract_vlm_object_crop`: cut the target object crop from the source frame.
- `build_vlm_frame_cropper_package`: create the crop package for VLM inference.
- `initialize_vlm_crop_cache`: prepare the local per-track candidate cache used before one-shot VLM dispatch.
- `update_vlm_crop_cache`: add or refresh one candidate crop for a track.
- `build_vlm_dispatch_package`: choose and emit the one crop that should actually be sent to VLM.
- `register_vlm_ack_package`: update cropper-side dispatch state using VLM acknowledgement feedback.

Suggested external packages:

- `opencv-python`: practical for crop extraction and image manipulation.
- `pillow`: useful when the downstream VLM interface expects PIL image objects.

Config parameters used:

- `config_vlm_enabled`: controls whether crop requests should be prepared for the VLM path.
- `config_vlm_crop_feedback_enabled`: when true, cropper may reopen collection after a VLM retry request; when false, the first dispatched crop completes that track.
- `config_vlm_crop_cache_size`: controls how many candidate crops are collected per track before the cropper selects one for dispatch.
- `config_vlm_dead_after_lost_frames`: controls how many consecutive `lost` states are tolerated before the cropper marks the track `dead`.

#### vlm_frame_cropper_request_package

Produces:

- `vlm_frame_cropper_request_package`

Package fields:

- `vlm_frame_cropper_frame_id`: frame identifier used to locate the source image.
- `vlm_frame_cropper_track_id`: track identifier for the object to crop.
- `vlm_frame_cropper_bbox`: crop bounds for the requested object.
- `vlm_frame_cropper_trigger_reason`: reason the object was sent for semantic analysis.

#### vlm_frame_cropper_layer_package

Produces:

- `vlm_frame_cropper_layer_package`

Package fields:

- `vlm_frame_cropper_layer_track_id`: track identifier carried into the VLM stage.
- `vlm_frame_cropper_layer_image`: cropped object image used for semantic analysis.
- `vlm_frame_cropper_layer_bbox`: object crop bounds associated with the crop image.

#### vlm_dispatch_package

Produces:

- `vlm_dispatch_package`

Package fields:

- `vlm_dispatch_track_id`: track identifier chosen for VLM dispatch.
- `vlm_dispatch_mode`: whether this is the initial, retry, or final best-available dispatch.
- `vlm_dispatch_reason`: reason the cropper decided to dispatch now.
- `vlm_dispatch_cached_crop_count`: number of candidate crops considered when dispatching.
- `vlm_frame_cropper_layer_package`: the selected crop package forwarded to the VLM layer.

Recommended `vlm_dispatch_mode` values in the current branch:

- `initial_candidate`
- `retry_candidate`
- `use_previous_sent_final`
- `dead_best_available`

#### vlm_frame_cropper_node

Purpose:

- Create an object-level crop for semantic analysis.
- Maintain the local crop cache and pick the one crop that should actually be sent to VLM.

Internal functions:

- `resolve_source_frame`: retrieve the source frame associated with the crop request.
- `crop_object_from_frame`: extract the object image using the requested bounding box.
- `validate_crop_result`: check that the crop is usable before VLM inference.
- `score_vlm_crop_candidate`: rank cached crops using detector confidence, crop area, and recency.
- `select_best_vlm_crop_candidate`: choose the current best crop from the local cache.

Current dead-track rule:

- if a track remains `lost` for `config_vlm_dead_after_lost_frames`, the cropper marks it terminal `dead`
- if the first cache round never reached `config_vlm_crop_cache_size`, the cropper still dispatches the best available partial-cache crop once the track is dead
- after a retry request, a dead track may instead finalize by reusing the previous sent image

Interacts with:

- `vlm_frame_cropper_layer`

### 8. VLM Layer

Purpose:

- Add semantic enrichment beyond detector classes.

Interacts with:

- `configuration_layer`
- `vlm_frame_cropper_layer`
- `vehicle_state_layer`
- `metadata_output_layer`
- `evaluation_output_layer`

Public functions:

- `initialize_vlm_layer`: load the configured VLM model.
- `run_vlm_inference`: perform semantic inference on the object crop.
- `normalize_vlm_result`: convert raw model output into the standard VLM format.
- `build_vlm_layer_package`: create the VLM result package for downstream layers.
- `build_vlm_ack_package`: create the acknowledgement package sent back to the cropper and vehicle-state layers after reviewing one dispatched crop.

Suggested external packages:

- `transformers`: common starting point for open-source VLM loading and inference.
- `vllm`: useful when serving larger transformer-based models efficiently, if compatible with the chosen VLM.
- `onnxruntime`: useful if the selected VLM is exported for portable inference.
- `tensorrt`: useful on Jetson-class hardware when the VLM path is optimized for deployment.

Config parameters used:

- `config_vlm_enabled`: controls whether the VLM layer is active.
- `config_vlm_model`: selects the VLM model loaded by the layer.
- `config_vlm_crop_feedback_enabled`: controls whether the VLM result may request a new crop round or whether the first dispatched crop must be accepted as final for that track.
- `config_device`: selects the execution target for VLM inference.

#### vlm_layer_package

Produces:

- `vlm_layer_package`

Package fields:

- `vlm_layer_track_id`: track identifier for the enriched object.
- `vlm_layer_query_type`: semantic question type applied to the crop.
- `vlm_layer_label`: primary semantic label returned by the VLM.
- `vlm_layer_attributes`: additional semantic attributes returned by the VLM.
- `vlm_layer_confidence`: confidence or certainty score for the semantic result.
- `vlm_layer_model_id`: identifier for the VLM model used for inference.

#### vlm_ack_package

Produces:

- `vlm_ack_package`

Package fields:

- `vlm_ack_track_id`: track identifier that the acknowledgement applies to.
- `vlm_ack_status`: acknowledgement decision, such as `accepted`, `retry_requested`, or `finalize_with_current`.
- `vlm_ack_reason`: short reason explaining the acknowledgement decision.
- `vlm_ack_retry_requested`: whether the cropper should reopen candidate selection for this track.

Recommended retry-reason vocabulary for `vlm_ack_reason` or the VLM-side `retry_reasons` field:

- `occluded`
- `bad_angle`

Current target gate rule:

- if VLM decides the crop is not one of the currently flagged detector labels, the acknowledgement is accepted with reason `no` and downstream state marks the track `no`
- if VLM decides the crop is one of the currently flagged detector labels, semantic JSON is accepted and downstream state marks the track `done`

Ownership:

- `vlm_layer` owns semantic inference output for the current query.
- `vlm_layer` owns the acknowledgement decision about whether the current crop is usable.
- `vlm_layer` does not own persistent object state.

#### vlm_inference_node

Purpose:

- Run semantic inference on the object crop.

Internal functions:

- `prepare_vlm_prompt`: prepare the semantic query or prompt for inference.
- `infer_vlm_semantics`: run model inference on the crop image.
- `parse_vlm_response`: extract structured semantic fields from the model response.

Expected `vehicle_semantics_v1` behavior:

- the prompt should first ask: is this one of the currently active YOLO labels, such as `truck` or `bus`?
- if no: return `is_truck=false`, `ack_status=accepted`, and no retry reasons
- if yes and the image is good enough: return rigid JSON with `wheel_count`, `ack_status=accepted`, and `retry_reasons=[]`
- if yes but the image is not good enough: return rigid JSON with `ack_status=retry_requested` plus one or more retry reasons such as `occluded` or `bad_angle`

Interacts with:

- `vlm_layer`

### 9. Metadata Output Layer

Purpose:

- Emit structured metadata as the main output product of the pipeline.

Interacts with:

- `configuration_layer`
- `vehicle_state_layer`
- `vlm_layer`
- `scene_awareness_layer`
- `ttn_export_layer`

Public functions:

- `build_metadata_output_layer_package`: assemble the structured metadata package.
- `serialize_metadata_output`: convert metadata into the selected output format.
- `emit_metadata_output`: send metadata to the configured output destination.

Suggested external packages:

- `pydantic`: useful for validating output schemas and structured records.
- `orjson`: useful for fast JSON serialization of metadata payloads.
- `pyyaml`: useful if metadata snapshots or configs are also emitted in YAML form.

Config parameters used:

- `config_metadata_output_enabled`: controls whether metadata output is emitted.

#### metadata_output_layer_package

Produces:

- `metadata_output_layer_package`

Package fields:

- `metadata_output_layer_timestamps`: timestamps associated with emitted metadata records.
- `metadata_output_layer_object_ids`: object identifiers included in the metadata output.
- `metadata_output_layer_classes`: stored or derived object classes included in the output.
- `metadata_output_layer_semantic_tags`: semantic labels and attributes included in the output.
- `metadata_output_layer_scene_tags`: optional scene-level labels included when scene awareness is enabled.
- `metadata_output_layer_counts`: aggregate counts included in the output.
- `metadata_output_layer_summaries`: higher-level event or frame summaries included in the output.

#### metadata_output_node

Purpose:

- Format and emit the final structured metadata package.

Internal functions:

- `merge_metadata_sources`: combine persistent state and semantic enrichment into one output view.
- `format_metadata_record`: convert merged data into the output schema.
- `write_metadata_record`: write or forward the formatted record.

Interacts with:

- `metadata_output_layer`

### 10. Evaluation Output Layer

Purpose:

- Collect benchmarking, debugging, and R and D telemetry.

Interacts with:

- `input_layer`
- `roi_layer`
- `yolo_layer`
- `tracking_layer`
- `vlm_layer`
- `scene_awareness_layer`

Public functions:

- `collect_evaluation_metrics`: gather metrics from active layers.
- `build_evaluation_output_layer_package`: create the telemetry package.
- `emit_evaluation_output`: write evaluation records to the configured sink.

Suggested external packages:

- `prometheus-client`: useful if runtime metrics should be exported in a monitoring-friendly format.
- `pandas`: useful for offline evaluation, metric aggregation, and experiment comparison.
- `mlflow`: useful if experiment tracking becomes part of the R and D workflow.

Config parameters used:

- `config_evaluation_output_enabled`: controls whether evaluation records are emitted.

#### evaluation_output_layer_package

Produces:

- `evaluation_output_layer_package`

Package fields:

- `evaluation_output_layer_fps`: measured end-to-end or stage-level throughput.
- `evaluation_output_layer_module_latency`: per-layer timing measurements.
- `evaluation_output_layer_detection_count`: number of detections generated in the current interval.
- `evaluation_output_layer_track_count`: number of tracked objects in the current interval.
- `evaluation_output_layer_vlm_call_count`: number of VLM calls made in the current interval.
- `evaluation_output_layer_scene_call_count`: number of scene-awareness calls made in the current interval.

#### evaluation_output_node

Purpose:

- Aggregate and emit telemetry records for development and benchmarking.

Internal functions:

- `record_module_latency`: store timing for one layer execution step.
- `record_detection_metrics`: store detection and tracking volume metrics.
- `write_evaluation_record`: write or forward the aggregated telemetry record.

Interacts with:

- `evaluation_output_layer`

### 11. Scene Awareness Layer

Purpose:

- Provide optional full-frame semantic reasoning outside the object-crop path.

Interacts with:

- `configuration_layer`
- `input_layer`
- `metadata_output_layer`
- `evaluation_output_layer`

Public functions:

- `initialize_scene_awareness_layer`: load the configured scene-awareness resources.
- `run_scene_awareness_inference`: perform full-frame semantic analysis.
- `build_scene_awareness_layer_package`: create the scene-level semantic package.

Suggested external packages:

- `transformers`: practical starting point for full-frame semantic inference with open models.
- `onnxruntime`: useful if the scene-awareness model is exported for portable inference.
- `tensorrt`: useful on Jetson-class hardware when optimizing the scene-awareness path.

Config parameters used:

- `config_scene_awareness_enabled`: controls whether the scene-awareness path is active.
- `config_device`: selects the execution target for scene-awareness inference.

#### scene_awareness_layer_package

Produces:

- `scene_awareness_layer_package`

Package fields:

- `scene_awareness_layer_frame_id`: frame identifier tied to the scene-level result.
- `scene_awareness_layer_timestamp`: timestamp tied to the scene-level result.
- `scene_awareness_layer_label`: primary scene-level semantic label.
- `scene_awareness_layer_attributes`: additional scene-level semantic attributes.
- `scene_awareness_layer_confidence`: confidence associated with the scene-level result.

#### scene_awareness_node

Purpose:

- Run optional full-frame semantic analysis.

Internal functions:

- `prepare_scene_awareness_input`: prepare the full-frame image for inference.
- `infer_scene_awareness`: run scene-level inference.
- `parse_scene_awareness_result`: convert raw inference output into the standard scene package.

Interacts with:

- `scene_awareness_layer`

### 12. TTN Export Layer

Purpose:

- Forward selected metadata to a future downstream export target.

Interacts with:

- `metadata_output_layer`

Public functions:

- `build_ttn_export_layer_package`: create the export-ready package.
- `serialize_ttn_payload`: convert metadata into the required TTN payload format.
- `emit_ttn_export`: send the payload to the external endpoint when enabled.

Suggested external packages:

- `paho-mqtt`: practical if the downstream transport uses MQTT.
- `requests`: useful if export is implemented over HTTP APIs instead of MQTT.

Config parameters used:

- No direct v1 config parameters are defined yet for this placeholder export layer.

#### ttn_export_layer_package

Produces:

- `ttn_export_layer_package`

Package fields:

- `ttn_export_layer_payload`: metadata payload prepared for external transmission.
- `ttn_export_layer_timestamp`: time when the export package was generated.
- `ttn_export_layer_target`: downstream export destination identifier.

#### ttn_export_node

Purpose:

- Send export-ready metadata to the external target when that path is implemented.

Internal functions:

- `prepare_ttn_message`: build the final outbound TTN message.
- `send_ttn_message`: transmit the message to the downstream system.
- `record_ttn_export_result`: store success or failure status for the export attempt.

Interacts with:

- `ttn_export_layer`

Important note:

- This remains a placeholder layer, not an implemented core path.

## Main Interaction Paths

### Main Runtime Path

`input_layer_package -> roi_layer_package -> yolo_layer_package -> tracking_layer_package -> vehicle_state_layer_package -> metadata_output_layer_package`

### Optional Object Enrichment Path

`tracking_layer_package -> vlm_frame_cropper_request_package -> vlm_frame_cropper_layer_package -> vlm_dispatch_package -> vlm_layer_package -> vehicle_state_layer_package -> metadata_output_layer_package`

### Optional VLM Ack Feedback Path

`vlm_ack_package -> vlm_frame_cropper_layer + vehicle_state_layer`

When `config_vlm_crop_feedback_enabled` is `false`:

- cropper still waits for one full cache round
- one best crop is dispatched once
- VLM returns one rigid JSON decision
- that track is marked progressed and no longer re-enters cropper or VLM, although tracking may continue updating it

When a track becomes dead before its first cache round is full:

- cropper may still emit one `dead_best_available` dispatch using the best available partial cache
- VLM performs a final target-label gate on that crop
- rejected-label results mark the track `no`; accepted flagged-label JSON marks the track `done`

### Optional Scene Path

`input_layer_package -> scene_awareness_layer_package -> metadata_output_layer_package`

### Optional Scene Evaluation Path

`scene_awareness_layer_package -> evaluation_output_layer_package`

### Optional Export Path

`metadata_output_layer_package -> ttn_export_layer_package`

## Ownership Summary

### tracking_layer owns

- object identity continuity
- new vs existing decision
- tracked-object lifecycle status

### vehicle_state_layer owns

- persistent object metadata
- first seen and last seen history
- stored semantic fields like truck type
- whether VLM enrichment has already been recorded

### vlm_layer owns

- semantic inference output for the current query
- acknowledgement of whether the current crop is accepted, needs retry, or must be finalized with the current best crop

### metadata_output_layer owns

- formatting and emitting structured downstream metadata

## Recommended Next Step

The next useful document should be a strict layer-to-layer interface table with columns like:

- source layer
- destination layer
- package name
- function that emits the package
- required fields
- optional fields
- notes



