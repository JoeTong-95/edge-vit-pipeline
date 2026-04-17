# YOLO Tag Filter Behavior

This document explains the difference between:

- what the bundled YOLO weights know how to detect
- what this repository currently forwards downstream

## Short Version

The bundled YOLO models in `src/yolo-layer/models/` are COCO-style pretrained detectors with many labels.

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

## Current Forwarded Tags

The current `TARGET_CLASSES` map is:

```python
TARGET_CLASSES = {
    2: "car",
    7: "truck",
    5: "bus",
}
```

These are the only classes currently forwarded downstream.

## Current Practical Meaning

- `truck`: large trucks that COCO-style YOLO labels as truck
- `bus`: large vehicles that YOLO labels as bus
- `car`: generic road-vehicle bucket that often includes sedans, SUVs, pickups, vans, and other smaller vehicle shapes

This is why SUV-like vehicles may appear as `car` instead of a finer subtype.

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

And, more generally, every COCO label except `car`, `bus`, and `truck`.

## How To Change Behavior

Edit `src/yolo-layer/class_map.py`.

Rules:

- adding a class ID enables that class in downstream pipeline output
- removing a class ID disables that class in downstream pipeline output
- changing the mapped string changes the label text written into `yolo_detection_class`

Example:

```python
TARGET_CLASSES = {
    2: "car",
    5: "bus",
    7: "truck",
}
```

If you changed it to:

```python
TARGET_CLASSES = {
    7: "truck",
}
```

then only YOLO detections with COCO class ID `7` would survive filtering.

If you changed it to:

```python
TARGET_CLASSES = {
    2: "road_vehicle",
    5: "road_vehicle",
    7: "road_vehicle",
}
```

then all three classes would still be forwarded, but downstream code would see the shared label `road_vehicle`.

## Important Caveat

Changing `class_map.py` changes the project filter, not the underlying weight file taxonomy.

So:

- adding `2: "car"` does not teach the model a new class
- removing `2: "car"` does not remove that class from the weight file
- it only changes whether the pipeline keeps or discards that predicted class

## Recommendation

Treat `class_map.py` as the detector-policy switchboard for the repo.

If you want to know whether a tag is used in the current pipeline, check this file first.
