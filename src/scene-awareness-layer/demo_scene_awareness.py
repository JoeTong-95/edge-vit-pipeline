from __future__ import annotations

import time

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

from scene_awareness_layer import (
    initialize_scene_awareness_layer,
    run_scene_awareness_inference,
)


def _make_fake_frame(width: int = 320, height: int = 180):
    """
    Create a simple synthetic image:
    - bright gradient background
    - a darker rectangle region to add edges/contrast
    """
    if np is None:
        # Pure-python fallback: H x W x 3 (BGR-like) nested lists
        frame = []
        for y in range(height):
            row = []
            for x in range(width):
                v = int(255 * (x / max(1, width - 1)))
                row.append([v, v, v])
            frame.append(row)
        # add a dark block
        for y in range(height // 3, height // 3 * 2):
            for x in range(width // 3, width // 3 * 2):
                frame[y][x] = [30, 30, 30]
        return frame

    # Numpy version (uint8 image)
    x = np.linspace(0, 255, width, dtype=np.uint8)
    grad = np.tile(x, (height, 1))
    frame = np.dstack([grad, grad, grad])  # H,W,3
    frame[height // 3 : height // 3 * 2, width // 3 : width // 3 * 2, :] = 30
    return frame


def main():
    runtime = initialize_scene_awareness_layer(
        config_scene_awareness_enabled=True, config_device="auto"
    )

    input_layer_package = {
        "input_layer_frame_id": 1,
        "input_layer_timestamp": time.time(),
        "input_layer_image": _make_fake_frame(),
        "input_layer_source_type": "demo",
        "input_layer_resolution": (320, 180),
    }

    scene_pkg = run_scene_awareness_inference(runtime, input_layer_package)
    print("scene_awareness_layer_package:")
    print(scene_pkg)


if __name__ == "__main__":
    main()

