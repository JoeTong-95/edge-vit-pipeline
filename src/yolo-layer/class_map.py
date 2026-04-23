# class_map.py
# Maps YOLO class IDs to project label names.
#
# Only classes listed here survive detection filtering.
# Everything else YOLO detects gets discarded.
#
# ACTIVE MODEL:
# - This active map is specific to the 6-class YOLOv11n fine-tune stored under
#   src/yolo-layer/models/ and deployed as
#   src/yolo-layer/models/yolo11n_6class_finetuned.engine.
# - That checkpoint uses a compact vehicle taxonomy tuned for traffic scenes.
#
# WHY THESE ACTIVE CLASSES:
# - Keep truck-like classes plus bus/van so the downstream truck/VLM pipeline
#   receives the detailed vehicle categories emitted by this custom checkpoint.
# - Class 0 ("car") remains disabled for now to avoid forwarding every normal
#   passenger car as a truck candidate.
#
# YOLO11N 6-CLASS IDS:
#   0 = car
#   1 = pickup
#   2 = van
#   3 = truck
#   4 = bus
#   5 = motorcycle
#
# PREVIOUS COCO MAP FOR REFERENCE:
# - Class 7 ("truck"): semis, box trucks, dump trucks, and other large trucks.
# - Class 5 ("bus"): some large commercial vehicles are labeled as bus.
# - Class 2 ("car"): COCO folds SUVs, pickups, vans, and many smaller
#   truck-ish road vehicles into the generic "car" label.
#
# This gives broader coverage for road vehicles that may matter to the
# downstream truck/VLM pipeline, even when the base detector does not expose
# a finer-grained SUV/pickup/van taxonomy.
#
# FULL COCO VEHICLE IDS FOR REFERENCE:
#   1 = bicycle
#   2 = car
#   3 = motorcycle
#   5 = bus
#   6 = train
#   7 = truck
#
TARGET_CLASSES = {
    0: "car",
    1: "pickup",
    2: "van",
    3: "truck",
    4: "bus",
}
