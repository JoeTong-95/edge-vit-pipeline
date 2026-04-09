# detector.py
# Layer 4: YOLO Detection
#
# Lives in: src/yolo-layer/
#
# Public functions (from pipeline_layers_and_interactions.md):
#   initialize_yolo_layer  - load the configured YOLO model
#   run_yolo_detection     - run detection on the active frame
#   filter_yolo_detections - remove detections below confidence threshold
#   build_yolo_layer_package - create the detection package for downstream tracking
#
# Produces: yolo_layer_package
#
# Expected upstream input: roi_layer_package (or input_layer_package if ROI disabled)
# Downstream consumer: tracking_layer

from pathlib import Path

from ultralytics import YOLO
from class_map import TARGET_CLASSES


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
# The YOLO model and config are loaded once at startup and reused every frame.
# This avoids reloading the model per frame (which would be very slow).

_state = {
    "model": None,
    "conf_threshold": None,
    "target_class_ids": None,
    "device": None,
    "initialized": False,
}

_LAYER_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _LAYER_DIR / "models"


# ---------------------------------------------------------------------------
# Public functions (spec-defined)
# ---------------------------------------------------------------------------

def initialize_yolo_layer(model_name="yolov8n.pt", conf_threshold=0.25, device="cpu"):
    """
    Load the configured YOLO model and store layer settings.

    This should be called once at pipeline startup, not per frame.
    Loading a model takes ~1-2 seconds; inference is fast after that.

    Args:
        model_name: Pretrained model name (e.g., "yolov8n.pt") or path
                    to a custom .pt file. The layer prefers bundled local
                    weights under src/yolo-layer/models before falling back
                    to Ultralytics lookup.
        conf_threshold: Minimum confidence to keep a detection (0.0 to 1.0).
                        Lower = more detections but more false positives.
                        0.25 is a reasonable starting default.
        device: Compute target. "cpu" for laptops, "cuda" for GPU,
                "cuda:0" for a specific GPU.
    """
    resolved_model_name = _resolve_model_path(model_name)
    _state["model"] = YOLO(resolved_model_name)
    _state["conf_threshold"] = conf_threshold
    _state["target_class_ids"] = set(TARGET_CLASSES.keys())
    _state["device"] = device
    _state["initialized"] = True

    print(f"[yolo_layer] Loaded model: {resolved_model_name}")
    print(f"[yolo_layer] Device: {device}")
    print(f"[yolo_layer] Confidence threshold: {conf_threshold}")
    print(f"[yolo_layer] Target classes: {TARGET_CLASSES}")


def run_yolo_detection(upstream_package):
    """
    Run YOLO detection on the image from the upstream package.

    Accepts either roi_layer_package or input_layer_package — it looks
    for the image field in either format and extracts it.

    Args:
        upstream_package: dict from ROI layer or input layer. Must contain
                          an image field (roi_layer_image or input_layer_image).

    Returns:
        raw_detections: list of dicts, each with:
            - "yolo_detection_bbox": [x1, y1, x2, y2]
            - "yolo_detection_class": str (project label from class_map)
            - "yolo_detection_class_id": int (original COCO ID, for debugging)
            - "yolo_detection_confidence": float
    """
    if not _state["initialized"]:
        raise RuntimeError("yolo_layer not initialized. Call initialize_yolo_layer() first.")

    image = upstream_package.get("roi_layer_image",
            upstream_package.get("input_layer_image"))

    if image is None:
        raise ValueError("Upstream package has no image field "
                         "(expected roi_layer_image or input_layer_image).")

    results = _state["model"](
        image,
        conf=_state["conf_threshold"],
        device=_state["device"],
        verbose=False,
    )

    result = results[0]

    raw_detections = []

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls.item())
            bbox = box.xyxy[0].tolist()
            confidence = round(float(box.conf.item()), 4)

            raw_detections.append({
                "yolo_detection_bbox": bbox,
                "yolo_detection_class": TARGET_CLASSES.get(class_id, f"class_{class_id}"),
                "yolo_detection_class_id": class_id,
                "yolo_detection_confidence": confidence,
            })

    return raw_detections


def filter_yolo_detections(raw_detections):
    """
    Remove detections that are below the confidence threshold or not
    in our target class list.

    run_yolo_detection already applies the confidence threshold at the
    model level, but this function provides a second filtering pass.
    This is useful if you want to post-filter at a different threshold
    or if you want to narrow the class list after detection.

    Args:
        raw_detections: list of detection dicts from run_yolo_detection.

    Returns:
        filtered_detections: list of detection dicts that passed all filters.
    """
    target_ids = _state["target_class_ids"]
    conf_threshold = _state["conf_threshold"]

    filtered = []
    for det in raw_detections:
        if det["yolo_detection_class_id"] not in target_ids:
            continue
        if det["yolo_detection_confidence"] < conf_threshold:
            continue
        filtered.append(det)

    return filtered


def build_yolo_layer_package(frame_id, filtered_detections):
    """
    Create the yolo_layer_package for downstream tracking.

    Args:
        frame_id: frame identifier from the upstream package
                  (roi_layer_frame_id or input_layer_frame_id).
        filtered_detections: list of detection dicts from filter_yolo_detections.

    Returns:
        yolo_layer_package: dict with:
            - "yolo_layer_frame_id": int
            - "yolo_layer_detections": list of detection dicts, each with:
                - "yolo_detection_bbox": [x1, y1, x2, y2]
                - "yolo_detection_class": str
                - "yolo_detection_confidence": float
    """
    clean_detections = []
    for det in filtered_detections:
        clean_detections.append({
            "yolo_detection_bbox": det["yolo_detection_bbox"],
            "yolo_detection_class": det["yolo_detection_class"],
            "yolo_detection_confidence": det["yolo_detection_confidence"],
        })

    return {
        "yolo_layer_frame_id": frame_id,
        "yolo_layer_detections": clean_detections,
    }


# ---------------------------------------------------------------------------
# Convenience function (not in the spec, but useful for clean pipeline code)
# ---------------------------------------------------------------------------

def process_frame(upstream_package):
    """
    Run the full YOLO pipeline on one frame: detect -> filter -> package.

    This is a convenience wrapper that calls the three public functions
    in sequence. The pipeline can call this instead of calling each
    function individually.

    Args:
        upstream_package: dict from ROI layer or input layer.

    Returns:
        yolo_layer_package: the standard detection package.
    """
    frame_id = upstream_package.get("roi_layer_frame_id",
               upstream_package.get("input_layer_frame_id"))

    raw_detections = run_yolo_detection(upstream_package)
    filtered_detections = filter_yolo_detections(raw_detections)
    yolo_layer_package = build_yolo_layer_package(frame_id, filtered_detections)

    return yolo_layer_package


def _resolve_model_path(model_name):
    """Prefer bundled local YOLO weights before falling back to Ultralytics lookup."""
    candidate = Path(model_name)
    if candidate.exists():
        return str(candidate)

    bundled_candidate = _MODEL_DIR / model_name
    if bundled_candidate.exists():
        return str(bundled_candidate)

    if candidate.suffix == "":
        bundled_pt_candidate = _MODEL_DIR / f"{model_name}.pt"
        if bundled_pt_candidate.exists():
            return str(bundled_pt_candidate)

    return model_name
