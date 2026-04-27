#!/usr/bin/env python3
"""GRACE Inference — Standalone script for FHWA vehicle classification.

Usage:
    # Single image
    python inference.py path/to/truck_crop.jpg

    # Multiple images
    python inference.py img1.jpg img2.jpg img3.jpg

    # Directory of images
    python inference.py --dir path/to/crops/

    # Batch with JSON output
    python inference.py --dir path/to/crops/ --output results.json

    # Use CPU
    python inference.py --device cpu path/to/truck_crop.jpg
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

from grace.models.axle_graph_v26 import AxleGraphModelV26
from grace.losses.ordinal import predict_from_ordinal_logits

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Label Maps ────────────────────────────────────────────────────
FHWA_NAMES = {
    0: "FHWA-2", 1: "FHWA-3", 2: "FHWA-4", 3: "FHWA-5",
    4: "FHWA-6", 5: "FHWA-7", 6: "FHWA-8", 7: "FHWA-9",
    8: "FHWA-10", 9: "FHWA-11", 10: "FHWA-12", 11: "FHWA-13",
    12: "FHWA-14",
}

VEHICLE_TYPES = {
    0: "sedan", 1: "suv", 2: "pickup", 3: "van", 4: "bus",
    5: "box_truck", 6: "dump_truck", 7: "flatbed", 8: "tanker",
    9: "concrete_mixer", 10: "garbage_truck", 11: "tow_truck",
    12: "car_carrier", 13: "logging_truck", 14: "semi_tractor",
    15: "delivery_van", 16: "utility_truck", 17: "recreational_vehicle",
    18: "emergency_vehicle", 19: "construction_vehicle", 20: "other",
}

TRAILER_NAMES = {0: "0", 1: "1", 2: "2", 3: "3+"}


def load_model(
    config_path: str = None,
    checkpoint_path: str = None,
    device: str = "cuda",
):
    """Load GRACE model from config + checkpoint.

    Args:
        config_path: Path to config.yaml (default: config.yaml next to this script)
        checkpoint_path: Path to .pt checkpoint (default: checkpoint/ dir)
        device: 'cuda' or 'cpu'

    Returns:
        (model, device)
    """
    if config_path is None:
        config_path = SCRIPT_DIR / "config.yaml"
    if checkpoint_path is None:
        checkpoint_path = SCRIPT_DIR / "checkpoint" / "best_axle_graph_v6.pt"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_config = config["model"].copy()
    model_config.pop("type", None)

    if "num_vehicle_types" in model_config and "num_primary_classes" not in model_config:
        model_config["num_primary_classes"] = model_config.pop("num_vehicle_types")
    if "backbone" in model_config and "backbone_name" not in model_config:
        model_config["backbone_name"] = model_config.pop("backbone")

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = AxleGraphModelV26(**model_config)

    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("projection_head."):
            continue
        new_state_dict[k[6:] if k.startswith("model.") else k] = v

    model.load_state_dict(new_state_dict)
    model.eval().to(dev)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"GRACE loaded: {n_params:.1f}M params on {dev}")
    return model, dev


def preprocess(image_path: str) -> torch.Tensor:
    """Load and preprocess a single truck crop → [1, 3, 224, 224]."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def predict_single(model, device, image_path: str) -> dict:
    """Run GRACE on one image, return structured results."""
    img = preprocess(image_path).to(device)

    with torch.no_grad():
        outputs = model(img, return_heatmap=False)

    fhwa_idx = outputs["fhwa_logits"].argmax(dim=-1).item()
    primary_idx = outputs["primary_logits"].argmax(dim=-1).item()
    trailer_idx = outputs["trailer_logits"].argmax(dim=-1).item()
    axle_count = predict_from_ordinal_logits(
        outputs["axle_count_logits"], method="expectation"
    ).item()

    fhwa_probs = torch.softmax(outputs["fhwa_logits"], dim=-1)[0]
    fhwa_conf = fhwa_probs[fhwa_idx].item()

    return {
        "image": str(image_path),
        "fhwa_class": FHWA_NAMES.get(fhwa_idx, f"IDX-{fhwa_idx}"),
        "fhwa_index": fhwa_idx,
        "fhwa_confidence": round(fhwa_conf, 4),
        "vehicle_type": VEHICLE_TYPES.get(primary_idx, f"IDX-{primary_idx}"),
        "vehicle_type_index": primary_idx,
        "trailer_count": TRAILER_NAMES.get(trailer_idx, str(trailer_idx)),
        "axle_count": round(axle_count, 2),
    }


def predict_batch(model, device, image_paths: list, batch_size: int = 16) -> list:
    """Run GRACE on a batch of images."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        tensors = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(transform(img))
            except Exception as e:
                results.append({"image": str(p), "error": str(e)})
                continue

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            outputs = model(batch, return_heatmap=False)

        fhwa_indices = outputs["fhwa_logits"].argmax(dim=-1)
        primary_indices = outputs["primary_logits"].argmax(dim=-1)
        trailer_indices = outputs["trailer_logits"].argmax(dim=-1)
        axle_counts = predict_from_ordinal_logits(
            outputs["axle_count_logits"], method="expectation"
        )
        fhwa_probs = torch.softmax(outputs["fhwa_logits"], dim=-1)

        for j, p in enumerate(batch_paths):
            if j >= len(tensors):
                break
            fi = fhwa_indices[j].item()
            results.append({
                "image": str(p),
                "fhwa_class": FHWA_NAMES.get(fi, f"IDX-{fi}"),
                "fhwa_index": fi,
                "fhwa_confidence": round(fhwa_probs[j, fi].item(), 4),
                "vehicle_type": VEHICLE_TYPES.get(primary_indices[j].item(), "unknown"),
                "vehicle_type_index": primary_indices[j].item(),
                "trailer_count": TRAILER_NAMES.get(trailer_indices[j].item(), "?"),
                "axle_count": round(axle_counts[j].item(), 2),
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="GRACE: FHWA Vehicle Classification")
    parser.add_argument("images", nargs="*", help="Image path(s)")
    parser.add_argument("--dir", type=str, help="Directory of images")
    parser.add_argument("--output", "-o", type=str, help="Save results to JSON file")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for directory mode")
    args = parser.parse_args()

    image_paths = list(args.images) if args.images else []
    if args.dir:
        d = Path(args.dir)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            image_paths.extend(sorted(d.glob(ext)))

    if not image_paths:
        parser.print_help()
        sys.exit(1)

    model, device = load_model(args.config, args.checkpoint, args.device)

    t0 = time.time()
    if len(image_paths) == 1:
        results = [predict_single(model, device, image_paths[0])]
    else:
        results = predict_batch(model, device, image_paths, args.batch_size)
    elapsed = time.time() - t0

    for r in results:
        if "error" in r:
            print(f"  ERROR {r['image']}: {r['error']}")
        else:
            print(
                f"  {r['fhwa_class']:>8s} ({r['fhwa_confidence']:.2f})  "
                f"type={r['vehicle_type']:<22s}  "
                f"axles={r['axle_count']:.1f}  "
                f"trailers={r['trailer_count']}  "
                f"| {Path(r['image']).name}"
            )

    print(f"\n{len(results)} images in {elapsed:.2f}s ({len(results)/max(elapsed,0.001):.1f} img/s)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
