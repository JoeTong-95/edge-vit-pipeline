#!/usr/bin/env bash
# export-yolo-trt.sh
#
# Converts every .pt YOLO weight in src/yolo-layer/models/ to a TensorRT FP16
# engine (.engine) suitable for Jetson inference.
#
# Run this INSIDE the vision-jetson container:
#   docker/run-docker-jetson   (opens a shell)
#   bash /app/docker/export-yolo-trt.sh
#
# Or non-interactively from the host:
#   docker run --rm --runtime=nvidia --ipc=host \
#       --ulimit memlock=-1 --ulimit stack=67108864 \
#       -v "$(pwd)":/app vision-jetson:latest \
#       bash /app/docker/export-yolo-trt.sh
#
# Output:  src/yolo-layer/models/<name>.engine  (FP16, optimised for current GPU)
#
# Notes:
#   - First export of a model takes ~5-15 min per model (TRT engine building).
#   - Engines are device-specific; rebuild if you change hardware.
#   - imgsz 640 matches the default Ultralytics inference size and the
#     config_frame_resolution width used in this project.
#   - The Jingtao model uses imgsz 640 as well; its ROI crops will be
#     dynamic-shape anyway once the pipeline is running.

set -euo pipefail

MODELS_DIR="/app/src/yolo-layer/models"
IMGSZ=640
BATCH=1

echo "=== YOLO → TensorRT FP16 export ==="
echo "Models dir : ${MODELS_DIR}"
echo "imgsz      : ${IMGSZ}"
echo "batch      : ${BATCH}"
echo ""

exported=0
skipped=0
failed=0

for pt_file in "${MODELS_DIR}"/*.pt; do
    [[ -f "$pt_file" ]] || continue
    base="${pt_file%.pt}"
    engine_file="${base}.engine"

    if [[ -f "$engine_file" ]]; then
        echo "[SKIP] $(basename "$engine_file") already exists"
        skipped=$((skipped + 1))
        continue
    fi

    echo "[EXPORT] $(basename "$pt_file") → $(basename "$engine_file") ..."
    start_ts=$(date +%s)

    python3 - <<PYEOF
from ultralytics import YOLO
import sys

model_path = "${pt_file}"
print(f"  Loading {model_path} ...")
model = YOLO(model_path)

print(f"  Exporting to TensorRT FP16 (imgsz=${IMGSZ}, batch=${BATCH}) ...")
export_path = model.export(
    format="engine",
    imgsz=${IMGSZ},
    half=True,
    batch=${BATCH},
    workspace=4,          # GB of GPU workspace during TRT build
    verbose=False,
)
print(f"  Saved: {export_path}")
PYEOF

    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    echo "[DONE] $(basename "$pt_file") in ${elapsed}s"
    exported=$((exported + 1))
done

echo ""
echo "=== Export summary ==="
echo "  Exported : ${exported}"
echo "  Skipped  : ${skipped} (already existed)"
echo "  Failed   : ${failed}"
echo ""
echo "Engines written to: ${MODELS_DIR}/"
ls -lh "${MODELS_DIR}"/*.engine 2>/dev/null || echo "(none found)"
