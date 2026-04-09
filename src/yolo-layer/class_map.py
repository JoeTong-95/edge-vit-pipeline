# class_map.py
# Maps COCO dataset class IDs to project label names.
#
# Only classes listed here survive detection filtering.
# Everything else YOLO detects gets discarded.
#
# WHY THESE THREE CLASSES:
# - Class 7 ("truck"): semis, box trucks, large commercial trucks.
# - Class 5 ("bus"): large delivery trucks sometimes get classified as bus.
#
# This gives broad coverage for "all kinds of trucks."
# Remove "car" or "bus" later once you see real detection results
# and decide what counts as a truck for your project.
#
# FULL COCO VEHICLE IDS FOR REFERENCE:
#   1 = bicycle
#   2 = car
#   3 = motorcycle
#   5 = bus
#   6 = train
#   7 = truck

TARGET_CLASSES = {
    7: "truck",
    5: "bus",
}
