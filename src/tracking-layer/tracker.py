# tracker.py
# Layer 5: Tracking
#
# Lives in: src/tracking-layer/
#
# Public functions (from pipeline_layers_and_interactions.md):
#   initialize_tracking_layer  - prepare the tracking algorithm state
#   update_tracks              - associate current detections with existing tracks
#   assign_tracking_status     - label each object as new, active, or lost
#   build_tracking_layer_package - create the tracking package for downstream layers
#
# Produces: tracking_layer_package
#
# Expected upstream input: yolo_layer_package
# Downstream consumers: vehicle_state_layer, vlm_frame_cropper_layer,
#                        metadata_output_layer
#
# DESIGN NOTES:
# - Uses supervision's ByteTrack, which accepts raw detections as input.
#   This keeps YOLO (Layer 4) and Tracking (Layer 5) truly separate.
# - Status logic (new/active/lost) is managed here, not by ByteTrack.
#   ByteTrack handles matching; we handle lifecycle labeling.
# - Lost tracks are emitted with their last known bbox for a configurable
#   number of frames, then dropped entirely.

import numpy as np
import supervision as sv


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_state = {
    "tracker": None,           # supervision ByteTrack instance
    "initialized": False,
    "max_lost_frames": 30,     # how long to keep reporting lost tracks

    # Track lifecycle bookkeeping
    "ever_seen_ids": set(),    # all track IDs ever encountered
    "prev_active_ids": set(),  # track IDs that were active/new last frame
    "track_history": {},       # track_id -> last known info dict
    "lost_counts": {},         # track_id -> frames since last seen
}


# ---------------------------------------------------------------------------
# Public functions (spec-defined)
# ---------------------------------------------------------------------------

def initialize_tracking_layer(max_lost_frames=30, track_activation_threshold=0.25,
                               frame_rate=30):
    """
    Prepare the ByteTrack tracker state.

    Call this once at pipeline startup.

    Args:
        max_lost_frames: How many frames a lost track stays in the output
                         before being dropped entirely. Default 30 (~1 second
                         at 30fps).
        track_activation_threshold: Minimum detection confidence for ByteTrack
                                     to consider creating a new track.
        frame_rate: Expected video frame rate. ByteTrack uses this internally
                    for motion prediction.
    """
    _state["tracker"] = sv.ByteTrack(
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=max_lost_frames,
        minimum_matching_threshold=0.8,
        frame_rate=frame_rate,
    )
    _state["max_lost_frames"] = max_lost_frames
    _state["ever_seen_ids"] = set()
    _state["prev_active_ids"] = set()
    _state["track_history"] = {}
    _state["lost_counts"] = {}
    _state["initialized"] = True

    print(f"[tracking_layer] Initialized ByteTrack")
    print(f"[tracking_layer] Max lost frames: {max_lost_frames}")
    print(f"[tracking_layer] Track activation threshold: {track_activation_threshold}")
    print(f"[tracking_layer] Frame rate: {frame_rate}")


def update_tracks(yolo_layer_package):
    """
    Feed current frame detections into ByteTrack and get track associations.

    This function converts the yolo_layer_package into the format ByteTrack
    expects, runs the tracker update, and returns the raw tracked results
    (before status labeling).

    Args:
        yolo_layer_package: dict with:
            - "yolo_layer_frame_id": int
            - "yolo_layer_detections": list of detection dicts

    Returns:
        current_tracks: list of dicts, each with:
            - "track_id": int (persistent ID from ByteTrack)
            - "bbox": [x1, y1, x2, y2]
            - "class": str (passed through from YOLO)
            - "confidence": float
    """
    if not _state["initialized"]:
        raise RuntimeError("tracking_layer not initialized. "
                           "Call initialize_tracking_layer() first.")

    detections_list = yolo_layer_package["yolo_layer_detections"]

    # Handle empty frames — still need to update the tracker so it
    # can age out lost tracks internally.
    if len(detections_list) == 0:
        empty_det = sv.Detections.empty()
        _state["tracker"].update_with_detections(empty_det)
        return []

    # Convert yolo_layer_package detections into numpy arrays for supervision
    bboxes = []
    confidences = []
    class_names = []  # we carry string labels through, not class IDs

    for det in detections_list:
        bboxes.append(det["yolo_detection_bbox"])
        confidences.append(det["yolo_detection_confidence"])
        class_names.append(det["yolo_detection_class"])

    bboxes_np = np.array(bboxes, dtype=np.float32)
    confidences_np = np.array(confidences, dtype=np.float32)

    # supervision.Detections needs class_id as int array.
    # We use a dummy 0-based index and carry the string class names separately.
    class_ids_np = np.zeros(len(bboxes), dtype=int)

    sv_detections = sv.Detections(
        xyxy=bboxes_np,
        confidence=confidences_np,
        class_id=class_ids_np,
    )

    # Run ByteTrack association
    tracked = _state["tracker"].update_with_detections(sv_detections)

    # Build our internal format
    current_tracks = []
    if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
        for i in range(len(tracked.tracker_id)):
            track_id = int(tracked.tracker_id[i])

            # Match back to the original class name.
            # After tracking, the order might change, so we use the
            # bbox overlap to find the best match. However, supervision
            # preserves the detection order after update, so index i
            # maps to the same detection.
            #
            # NOTE: supervision may drop some detections (low confidence
            # ones that ByteTrack didn't promote). The tracked detections
            # are a subset, but the xyxy values let us match back.
            # For simplicity, we use the confidence-sorted match.
            bbox = tracked.xyxy[i].tolist()
            confidence = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

            # Find matching class name by bbox proximity to original detections
            matched_class = _match_class_name(bbox, detections_list)

            current_tracks.append({
                "track_id": track_id,
                "bbox": bbox,
                "class": matched_class,
                "confidence": round(confidence, 4),
            })

    return current_tracks


def assign_tracking_status(current_tracks, frame_id):
    """
    Label each tracked object as 'new', 'active', or 'lost'.

    Rules:
        - 'new': track_id seen for the first time ever.
        - 'active': track_id was seen before and is still present.
        - 'lost': track_id was active previously but is not in current frame.

    Lost tracks are included in the output with their last known bbox
    for up to max_lost_frames frames, then dropped entirely.

    Args:
        current_tracks: list of track dicts from update_tracks.
        frame_id: current frame identifier.

    Returns:
        status_tracks: list of dicts, each with all track fields plus:
            - "status": "new", "active", or "lost"
    """
    current_ids = set(t["track_id"] for t in current_tracks)
    status_tracks = []

    # --- Process currently visible tracks ---
    for track in current_tracks:
        tid = track["track_id"]

        if tid not in _state["ever_seen_ids"]:
            # First time seeing this track
            status = "new"
            _state["ever_seen_ids"].add(tid)
        else:
            status = "active"

        # Update history with latest info
        _state["track_history"][tid] = {
            "bbox": track["bbox"],
            "class": track["class"],
            "confidence": track["confidence"],
            "last_frame": frame_id,
        }

        # Clear any lost counter since this track is visible
        if tid in _state["lost_counts"]:
            del _state["lost_counts"][tid]

        status_tracks.append({
            "track_id": tid,
            "bbox": track["bbox"],
            "class": track["class"],
            "confidence": track["confidence"],
            "status": status,
        })

    # --- Process lost tracks ---
    # Tracks that were active previously but not in current frame
    lost_ids = _state["prev_active_ids"] - current_ids

    # Also check tracks already marked lost from earlier frames
    for tid in list(_state["lost_counts"].keys()):
        if tid not in current_ids:
            lost_ids.add(tid)

    for tid in lost_ids:
        if tid in current_ids:
            continue  # not actually lost

        # Increment lost counter
        _state["lost_counts"][tid] = _state["lost_counts"].get(tid, 0) + 1

        # Drop if lost too long
        if _state["lost_counts"][tid] > _state["max_lost_frames"]:
            del _state["lost_counts"][tid]
            if tid in _state["track_history"]:
                del _state["track_history"][tid]
            continue

        # Emit lost track with last known info
        if tid in _state["track_history"]:
            history = _state["track_history"][tid]
            status_tracks.append({
                "track_id": tid,
                "bbox": history["bbox"],
                "class": history["class"],
                "confidence": history["confidence"],
                "status": "lost",
            })

    # Update previous active set for next frame
    _state["prev_active_ids"] = current_ids.copy()

    return status_tracks


def build_tracking_layer_package(frame_id, status_tracks):
    """
    Create the tracking_layer_package for downstream layers.

    Args:
        frame_id: current frame identifier.
        status_tracks: list of track dicts from assign_tracking_status.

    Returns:
        tracking_layer_package: dict with:
            - "tracking_layer_frame_id": int
            - "tracking_layer_tracks": list of dicts, each with:
                - "tracking_layer_track_id": int
                - "tracking_layer_bbox": [x1, y1, x2, y2]
                - "tracking_layer_detector_class": str
                - "tracking_layer_confidence": float
                - "tracking_layer_status": "new" | "active" | "lost"
    """
    formatted_tracks = []
    for track in status_tracks:
        formatted_tracks.append({
            "tracking_layer_track_id": track["track_id"],
            "tracking_layer_bbox": track["bbox"],
            "tracking_layer_detector_class": track["class"],
            "tracking_layer_confidence": track["confidence"],
            "tracking_layer_status": track["status"],
        })

    return {
        "tracking_layer_frame_id": frame_id,
        "tracking_layer_tracks": formatted_tracks,
    }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def process_frame(yolo_layer_package):
    """
    Run the full tracking pipeline on one frame: update -> status -> package.

    Args:
        yolo_layer_package: detection package from Layer 4.

    Returns:
        tracking_layer_package: the standard tracking package.
    """
    frame_id = yolo_layer_package["yolo_layer_frame_id"]

    current_tracks = update_tracks(yolo_layer_package)
    status_tracks = assign_tracking_status(current_tracks, frame_id)
    tracking_pkg = build_tracking_layer_package(frame_id, status_tracks)

    return tracking_pkg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _match_class_name(bbox, detections_list):
    """
    Find the class name from the original detections list that best matches
    a tracked bbox. Uses the closest bbox by center distance.

    This is needed because supervision's ByteTrack may reorder detections
    during association.
    """
    if not detections_list:
        return "unknown"

    bx = (bbox[0] + bbox[2]) / 2
    by = (bbox[1] + bbox[3]) / 2

    best_class = "unknown"
    best_dist = float("inf")

    for det in detections_list:
        db = det["yolo_detection_bbox"]
        dx = (db[0] + db[2]) / 2
        dy = (db[1] + db[3]) / 2
        dist = (bx - dx) ** 2 + (by - dy) ** 2

        if dist < best_dist:
            best_dist = dist
            best_class = det["yolo_detection_class"]

    return best_class
