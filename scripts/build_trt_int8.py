#!/usr/bin/env python3
"""
Build a TRT INT8 engine directly from an ONNX file using TensorRT Python API.
Uses torch tensors for GPU memory in the calibrator (no pycuda required).

Usage:
    python3 scripts/build_trt_int8.py \
        --onnx src/yolo-layer/models/yolov11v28_jingtao.onnx \
        --engine src/yolo-layer/models/yolov11v28_jingtao_int8.engine \
        --calib data/calib_frames \
        --imgsz 384 640
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class TorchInt8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 entropy calibrator using torch tensors for GPU buffers (no pycuda)."""

    def __init__(self, calib_dir, imgsz=(384, 640), batch_size=1, cache_file=None):
        super().__init__()
        self.h, self.w = imgsz
        self.batch_size = batch_size
        self.cache_file = cache_file or f"calib_int8_{self.h}x{self.w}.cache"
        self.images = sorted(Path(calib_dir).glob("*.jpg")) + \
                      sorted(Path(calib_dir).glob("*.png"))
        self.current_index = 0
        # Single pre-allocated GPU buffer — avoids repeated alloc/free during calibration
        self.device_input = torch.zeros(
            batch_size, 3, self.h, self.w, dtype=torch.float32, device="cuda"
        )
        print(f"[calibrator] {len(self.images)} images, imgsz=({self.h},{self.w}), "
              f"GPU buf={self.device_input.nbytes/1024:.0f} KB")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.images):
            return None
        img_path = self.images[self.current_index]
        img = cv2.imread(str(img_path))
        if img is None:
            self.current_index += 1
            return self.get_batch(names)
        img = cv2.resize(img, (self.w, self.h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to("cuda")
        self.device_input.copy_(img)
        if self.current_index % 50 == 0:
            print(f"[calibrator] batch {self.current_index}/{len(self.images)}")
        self.current_index += 1
        return [self.device_input.data_ptr()]

    def read_calibration_cache(self):
        if Path(self.cache_file).exists():
            print(f"[calibrator] loading cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"[calibrator] cache written to {self.cache_file}")


def build_int8_engine(onnx_path, engine_path, calib_dir, imgsz, workspace_gb=2):
    calibrator = TorchInt8Calibrator(
        calib_dir=calib_dir,
        imgsz=imgsz,
        batch_size=1,
        cache_file=str(Path(engine_path).with_suffix(".calib_cache")),
    )

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(
             1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
         ) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:

        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
        )
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator

        print(f"[trt_build] Parsing ONNX: {onnx_path}")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print("ONNX parse error:", parser.get_error(i))
                raise RuntimeError("ONNX parsing failed")

        print(f"[trt_build] Network: {network.num_layers} layers, "
              f"input={network.get_input(0).shape}")

        print("[trt_build] Building INT8 engine (calibration + kernel autotuning)...")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TRT engine build returned None — check TRT logs above")

        with open(engine_path, "wb") as f:
            f.write(serialized)

    engine_mb = Path(engine_path).stat().st_size / 1024**2
    print(f"[trt_build] Engine saved: {engine_path} ({engine_mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--imgsz", nargs=2, type=int, default=[384, 640],
                    metavar=("H", "W"))
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    print(f"ONNX:   {args.onnx}")
    print(f"Engine: {args.engine}")
    print(f"Calib:  {args.calib}")
    print(f"imgsz:  {args.imgsz[0]}x{args.imgsz[1]} (HxW)")

    Path(args.engine).parent.mkdir(parents=True, exist_ok=True)
    build_int8_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        calib_dir=args.calib,
        imgsz=tuple(args.imgsz),
        workspace_gb=args.workspace_gb,
    )


if __name__ == "__main__":
    main()
