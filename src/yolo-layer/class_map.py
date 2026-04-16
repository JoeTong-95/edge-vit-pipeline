# class_map.py
# Maps YOLO class IDs to project label names.
#
# Only classes listed here survive detection filtering.
# Everything else YOLO detects gets discarded.
#
# ACTIVE MODEL:
# - This active map is specific to src/yolo-layer/models/yolov11v28_jingtao.pt.
# - That checkpoint uses a custom vehicle taxonomy, not COCO IDs.
#
# WHY THESE ACTIVE CLASSES:
# - Keep truck-like classes plus bus/van so the downstream truck/VLM pipeline
#   receives the detailed vehicle categories emitted by this custom checkpoint.
# - Class 0 ("car") remains disabled for now to avoid forwarding every normal
#   passenger car as a truck candidate.
#
# CUSTOM yolov11v28_jingtao.pt IDS:
#   0 = car
#   1 = pickup_truck
#   2 = bus
#   3 = van
#   4 = tow_truck
#   5 = semi_truck
#   6 = box_truck
#   7 = dump_truck
#   8 = construction
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
# TARGET_CLASSES = {
#     # 2: "car",
#     7: "truck",
#     5: "bus",
# }

TARGET_CLASSES = {
    # 0: "car",
    1: "pickup_truck",
    2: "bus",
    3: "van",
    4: "tow_truck",
    5: "semi_truck",
    6: "box_truck",
    7: "dump_truck",
}
