from __future__ import annotations

from metadata_output_layer import (
    build_metadata_output_layer_package,
    emit_metadata_output,
    serialize_metadata_output,
)


def main() -> None:
    # Minimal fake vehicle_state_layer_package (parallel lists contract).
    vehicle_state_layer_package = {
        "vehicle_state_layer_track_id": [2, 1],
        "vehicle_state_layer_first_seen_frame": [10, 3],
        "vehicle_state_layer_last_seen_frame": [42, 42],
        "vehicle_state_layer_lost_frame_count": [0, 1],
        "vehicle_state_layer_vehicle_class": ["truck", "car"],
        "vehicle_state_layer_truck_type": ["box_truck", None],
        "vehicle_state_layer_semantic_tags": [["fleet_vehicle", "white"], ["sedan"]],
        "vehicle_state_layer_vlm_called": [True, False],
    }

    # Optional VLM enrichment (parallel lists contract).
    vlm_layer_package = {
        "vlm_layer_track_id": [1],
        "vlm_layer_query_type": ["truck_type"],
        "vlm_layer_label": ["compact_sedan"],
        "vlm_layer_attributes": [{"color": "red", "doors": 4}],
        "vlm_layer_confidence": [0.73],
        "vlm_layer_model_id": ["demo-model"],
    }

    # Optional scene awareness package (single-record dict contract).
    scene_awareness_layer_package = {
        "scene_awareness_layer_frame_id": "frame_0042",
        "scene_awareness_layer_timestamp": "2026-04-15T00:00:00Z",
        "scene_awareness_layer_label": "industrial_roadway",
        "scene_awareness_layer_attributes": ["daytime", "clear_weather"],
        "scene_awareness_layer_confidence": 0.66,
    }

    metadata_pkg = build_metadata_output_layer_package(
        vehicle_state_layer_package=vehicle_state_layer_package,
        vlm_layer_package=vlm_layer_package,
        scene_awareness_layer_package=scene_awareness_layer_package,
    )
    serialized = serialize_metadata_output(metadata_pkg, output_format="json")
    emit_metadata_output(serialized, output_destination="stdout")


if __name__ == "__main__":
    main()

