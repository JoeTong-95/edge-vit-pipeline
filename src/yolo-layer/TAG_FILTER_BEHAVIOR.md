# YOLO Tag Filter Behavior

This document explains the difference between:

- what the bundled YOLO weights know how to detect
- what this repository currently forwards downstream

## Short Version

The bundled YOLO models in `src/yolo-layer/models/` may know more labels than the pipeline actually uses.

The pipeline does **not** forward all of those labels.

Instead, the project applies a class filter in `class_map.py`, and only the IDs listed there survive into:

- `visualize_yolo.py`
- `visualize_tracking.py`
- `automated_evaluation.py`
- `visualize_roi.py`
- `visualize_vlm_frame_cropper.py`
- `visualize_vlm.py`
- `visualize_vlm_realtime.py`

That means commenting out, removing, or adding entries in `class_map.py` directly changes runtime detector behavior for the whole Python pipeline path.

## Current Target Tags

The current `TARGET_CLASSES` map is:

```python
TARGET_CLASSES = {
    1: "pickup",
    2: "van",
    3: "truck",
    4: "bus",
}
```

These are the only classes currently forwarded downstream.

## Current Practical Meaning

- `pickup`: pickup trucks
- `van`: vans and similar van-like vehicles
- `truck`: large trucks and truck-like heavy vehicles
- `bus`: buses and similar large passenger vehicles

These four labels are the current project target classes.

## What Will Not Be Forwarded Right Now

Any YOLO class not present in `TARGET_CLASSES` is discarded before building `yolo_layer_package`.

Examples of currently discarded COCO labels include:

- `person`
- `bicycle`
- `motorcycle`
- `airplane`
- `train`
- `boat`
- `traffic light`
- `stop sign`
- `dog`

And, more generally, every detector label outside `pickup`, `van`, `truck`, and `bus`.

## How To Change Behavior

Edit `src/yolo-layer/class_map.py`.

Rules:

- adding a class ID enables that class in downstream pipeline output
- removing a class ID disables that class in downstream pipeline output
- changing the mapped string changes the label text written into `yolo_detection_class`

Example:

```python
TARGET_CLASSES = {
    1: "pickup",
    2: "van",
    3: "truck",
    4: "bus",
}
```

If you changed it to:

```python
TARGET_CLASSES = {
    3: "truck",
}
```

then only detections with the model's `truck` class ID would survive filtering.

If you changed it to:

```python
TARGET_CLASSES = {
    1: "target_vehicle",
    2: "target_vehicle",
    3: "target_vehicle",
    4: "target_vehicle",
}
```

then all four target classes would still be forwarded, but downstream code would see the shared label `target_vehicle`.

## Important Caveat

Changing `class_map.py` changes the project filter, not the underlying weight file taxonomy.

So:

- adding a mapping does not teach the model a new class
- removing a mapping does not remove that class from the weight file
- it only changes whether the pipeline keeps or discards that predicted class

## Recommendation

Treat `class_map.py` as the detector-policy switchboard for the repo.

If you want to know whether a tag is used in the current pipeline, check this file first.
