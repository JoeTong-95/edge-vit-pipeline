# class_map.py
# Maps COCO dataset class IDs to project label names.
#
# Only classes listed here survive detection filtering.
# Everything else YOLO detects gets discarded.
#
# WHY THESE CLASSES:
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

TARGET_CLASSES = {
    #2: "car",
    7: "truck",
    5: "bus",
}
