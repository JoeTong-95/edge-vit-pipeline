#!/usr/bin/env python3
"""
Export SmolVLM-256M SigLIP vision encoder to ONNX, then build a TRT FP16 engine.

The SmolVLM-256M-Instruct vision encoder is a 12-layer SigLIP ViT (image_size=512,
patch_size=16, hidden_size=768).  It produces [batch, 1024, 768] patch features
that are downstream-consumed by the Idefics3 connector + LM.  This component has
a fixed spatial structure that TRT optimises very well (kernel fusion, FP16 tensor
cores, CUDA graph capture).

Usage (from repo root):
    python3 scripts/build_trt_smolvlm_vision.py \\
        --model src/vlm-layer/SmolVLM-256M-Instruct \\
        --engine src/vlm-layer/SmolVLM-256M-Instruct/vision_encoder_fp16.trt

Optional flags:
    --max-batch     Maximum batch size for the dynamic TRT profile (default: 5).
                    SmolVLM-256M splits images into at most scale_factor² + 1 = 17
                    sub-tiles; for typical vehicle crops with 1-4 tiles, 5 is enough.
    --workspace-gb  TRT builder workspace in GB (default: 2.0).
    --keep-onnx     Keep the intermediate ONNX file alongside the TRT engine.

After building, set  config_vlm_trt_vision_engine  in your Jetson YAML to the
engine path, or leave it at None and the VLM layer will auto-discover
vision_encoder_fp16.trt next to the model directory.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# SmolVLM-256M-Instruct vision encoder spatial constants.
_IMAGE_SIZE = 512          # SigLIP input resolution (H = W = 512)
_PATCH_SIZE = 16           # SigLIP patch stride
_NUM_PATCHES = (_IMAGE_SIZE // _PATCH_SIZE) ** 2   # 1024 patches per image
_VISION_HIDDEN_SIZE = 768  # SigLIP ViT hidden dimension


# ---------------------------------------------------------------------------
# ONNX export helpers
# ---------------------------------------------------------------------------

class _VisionEncoderWrapper(torch.nn.Module):
    """Wrap Idefics3VisionTransformer for clean ONNX tracing.

    The problem: Idefics3VisionTransformer.forward unconditionally calls
    create_bidirectional_mask() which converts the bool patch mask into SDPA
    format using dynamic shape ops — these break TorchScript tracing.

    The fix: monkey-patch create_bidirectional_mask to return None (= full
    attention) during tracing only.  For complete image tiles (no padding) full
    attention is mathematically identical to an all-ones mask, so the exported
    model produces identical outputs to the original at runtime.
    """

    def __init__(self, vision_model: Any) -> None:
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vision_model(pixel_values=pixel_values)
        return out[0]   # last_hidden_state [batch, num_patches, hidden]


def _patch_create_bidirectional_mask() -> tuple[Any, Any]:
    """Replace create_bidirectional_mask with a no-op that returns None.

    Returns (module, original_fn) so the caller can restore after tracing.
    None tells SDPA to use full (unmasked) attention — correct for non-padded tiles.
    """
    import transformers.models.idefics3.modeling_idefics3 as idefics3_mod

    original = idefics3_mod.create_bidirectional_mask

    def _noop_mask(*args: Any, **kwargs: Any) -> None:
        return None

    idefics3_mod.create_bidirectional_mask = _noop_mask
    return idefics3_mod, original


def export_vision_encoder_onnx(model_path: Path, onnx_path: Path) -> None:
    """Load SmolVLM on CPU, patch the tracing blocker, and export the vision encoder to ONNX.

    Keeping the model on CPU ensures the GPU is completely free when TRT builds
    the engine — critical on Jetson where the CUDA caching allocator doesn't
    release memory back to the OS even after del + empty_cache().
    """
    print(f"[export] Loading model on CPU from {model_path} …")

    from transformers import AutoModelForImageTextToText  # type: ignore[import-untyped]

    # CPU load: GPU stays free for TRT build; ONNX tracing works on CPU too.
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()   # stays on CPU

    vision_model = model.model.vision_model
    wrapper = _VisionEncoderWrapper(vision_model).eval()

    # CPU dummy input for tracing.
    dummy_input = torch.zeros(1, 3, _IMAGE_SIZE, _IMAGE_SIZE, dtype=torch.float32)

    # Warmup to verify correctness before patching anything.
    with torch.inference_mode():
        ref_out = wrapper(dummy_input)
    print(f"[export] Vision encoder warmup OK — output shape: {tuple(ref_out.shape)}")

    # Patch create_bidirectional_mask → None (full attention) for ONNX tracing only.
    # For complete image tiles there is no padding — full attention is identical.
    idefics3_mod, original_cbm = _patch_create_bidirectional_mask()
    print("[export] Patched create_bidirectional_mask → full-attention for tracing.")

    try:
        print(f"[export] Tracing vision encoder to ONNX: {onnx_path}")
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (dummy_input,),
                str(onnx_path),
                input_names=["pixel_values"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "pixel_values": {0: "batch"},
                    "last_hidden_state": {0: "batch"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
    finally:
        idefics3_mod.create_bidirectional_mask = original_cbm

    actual_patches = ref_out.shape[1]
    actual_hidden = ref_out.shape[2]
    print(
        f"[export] ONNX saved — "
        f"input [batch,3,{_IMAGE_SIZE},{_IMAGE_SIZE}] → "
        f"output [batch,{actual_patches},{actual_hidden}]"
    )

    # CPU model: no GPU memory to free.  Just drop references.
    del model, vision_model, wrapper


# ---------------------------------------------------------------------------
# TRT build helpers
# ---------------------------------------------------------------------------

def _fix_onnx_scatter_nd_type_mismatch(model: Any) -> Any:
    """Fix ScatterND nodes where the data tensor is Float but updates are Int64.

    Root cause in Idefics3VisionEmbeddings.forward: `position_ids` is created by
    `torch.full(..., fill_value=0)` which defaults to Float32, while `pos_ids`
    from `torch.bucketize` is Int64.  TRT 10 requires both to be the same type.

    Fix: cast the Float data tensor to Int64 before ScatterND.  This is also the
    correct type for the downstream nn.Embedding (Gather) which expects Long indices.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.shape_inference.infer_shapes(model)

    type_map: dict[str, int] = {}
    for vi in model.graph.value_info:
        type_map[vi.name] = vi.type.tensor_type.elem_type
    for inp in model.graph.input:
        type_map[inp.name] = inp.type.tensor_type.elem_type
    for init in model.graph.initializer:
        type_map[init.name] = init.data_type
    for out in model.graph.output:
        type_map[out.name] = out.type.tensor_type.elem_type

    new_nodes: list[Any] = []
    cast_count = 0
    fixed = 0

    for node in model.graph.node:
        if node.op_type == "ScatterND" and len(node.input) >= 3:
            data_name = node.input[0]
            updates_name = node.input[2]
            data_type = type_map.get(data_name, TensorProto.FLOAT)
            updates_type = type_map.get(updates_name, TensorProto.INT64)

            if data_type != updates_type:
                # Cast Float data → Int64 to match updates and fix the mismatch.
                cast_name = f"_cast_scatter_data_{cast_count}"
                cast_count += 1
                new_nodes.append(
                    helper.make_node("Cast", inputs=[data_name], outputs=[cast_name], to=updates_type)
                )
                new_nodes.append(
                    helper.make_node(
                        "ScatterND",
                        inputs=[cast_name, node.input[1], updates_name],
                        outputs=list(node.output),
                        name=node.name,
                    )
                )
                print(f"  [fix] ScatterND '{node.name}': cast data Float→Int64")
                fixed += 1
                continue
        new_nodes.append(node)

    if fixed == 0:
        return model  # no changes needed

    new_graph = helper.make_graph(
        new_nodes,
        model.graph.name,
        list(model.graph.input),
        list(model.graph.output),
        list(model.graph.initializer),
    )
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version
    return new_model


def _fix_onnx(onnx_path: Path) -> Path:
    """Run onnxsim then apply targeted ScatterND type fix. Returns path to fixed ONNX."""
    import onnx

    model = onnx.load(str(onnx_path))

    # Step 1: onnxsim (constant folding, dead code, etc.)
    try:
        import onnxsim  # type: ignore[import-untyped]

        simplified, ok = onnxsim.simplify(model)
        if ok:
            model = simplified
            print("[fix_onnx] onnxsim OK.")
        else:
            print("[fix_onnx] onnxsim returned not-OK; continuing with original.")
    except ModuleNotFoundError:
        print("[fix_onnx] onnxsim not installed; skipping.")

    # Step 2: Fix ScatterND Float/Int64 type mismatches.
    model = _fix_onnx_scatter_nd_type_mismatch(model)

    fixed_path = onnx_path.with_suffix(".fixed.onnx")
    onnx.save(model, str(fixed_path))
    print(f"[fix_onnx] Saved fixed ONNX: {fixed_path}")
    return fixed_path


def build_trt_fp16_engine(
    onnx_path: Path,
    engine_path: Path,
    max_batch: int,
    workspace_gb: float,
) -> None:
    """Build a TRT FP16 engine from the ONNX file with a dynamic batch profile."""
    # Fix type mismatches (ScatterND Float/Int64) before TRT parsing.
    onnx_path = _fix_onnx(onnx_path)

    with (
        trt.Builder(TRT_LOGGER) as builder,
        builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        ) as network,
        trt.OnnxParser(network, TRT_LOGGER) as parser,
        builder.create_builder_config() as config,
    ):
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30))
        )
        config.set_flag(trt.BuilderFlag.FP16)

        print(f"[trt_build] Parsing ONNX: {onnx_path}")
        with open(onnx_path, "rb") as f:
            raw = f.read()
        if not parser.parse(raw):
            for i in range(parser.num_errors):
                print("  ONNX parse error:", parser.get_error(i))
            raise RuntimeError("ONNX parsing failed — check errors above.")

        num_layers = network.num_layers
        in_shape = network.get_input(0).shape
        print(f"[trt_build] Network parsed: {num_layers} layers, input shape: {in_shape}")

        # On Jetson (NvMap unified memory), large contiguous tactic allocations fail
        # even with plenty of total free memory.  Cap the tactic DRAM pool so TRT
        # falls back to smaller (non-fusing) kernels that fit in Jetson's allocator.
        TACTIC_DRAM_LIMIT = 256 * 1024 * 1024  # 256 MB — avoids 343 MB NvMap failure
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.TACTIC_DRAM, TACTIC_DRAM_LIMIT)
        except Exception:
            pass  # older TRT versions may not have TACTIC_DRAM

        opt_batch = min(2, max_batch)
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "pixel_values",
            min=(1, 3, _IMAGE_SIZE, _IMAGE_SIZE),
            opt=(opt_batch, 3, _IMAGE_SIZE, _IMAGE_SIZE),
            max=(max_batch, 3, _IMAGE_SIZE, _IMAGE_SIZE),
        )
        config.add_optimization_profile(profile)

        if max_batch == 1:
            print(
                f"[trt_build] Building FP16 engine "
                f"(static batch=1, tactic_dram_limit=256MB) …"
            )
        else:
            print(
                f"[trt_build] Building FP16 engine "
                f"(batch 1–{max_batch}, opt={opt_batch}, workspace {workspace_gb:.1f} GB) …"
            )
        print("[trt_build] This may take several minutes on first build.")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TRT engine build returned None — inspect TRT logs above.")

        with open(engine_path, "wb") as f:
            f.write(serialized)

    engine_mb = engine_path.stat().st_size / 1024 ** 2
    print(f"[trt_build] Engine saved: {engine_path}  ({engine_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--model", required=True,
        help="Path to SmolVLM-256M-Instruct model directory.",
    )
    ap.add_argument(
        "--engine", required=True,
        help="Output path for the TRT FP16 engine file (.trt).",
    )
    ap.add_argument(
        "--max-batch", type=int, default=5,
        help="Maximum batch size for the TRT dynamic profile (default: 5).",
    )
    ap.add_argument(
        "--workspace-gb", type=float, default=2.0,
        help="TRT builder workspace in GB (default: 2.0).",
    )
    ap.add_argument(
        "--keep-onnx", action="store_true",
        help="Keep the intermediate ONNX file alongside the TRT engine.",
    )
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    model_path = Path(args.model).expanduser().resolve()
    engine_path = Path(args.engine).expanduser().resolve()

    if not model_path.is_dir():
        raise SystemExit(f"Model directory not found: {model_path}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model:      {model_path}")
    print(f"Engine:     {engine_path}")
    print(f"Max batch:  {args.max_batch}")
    print(f"Workspace:  {args.workspace_gb:.1f} GB")
    print()

    if args.keep_onnx:
        onnx_path = engine_path.with_suffix(".onnx")
        export_vision_encoder_onnx(model_path, onnx_path)
        build_trt_fp16_engine(onnx_path, engine_path, args.max_batch, args.workspace_gb)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "vision_encoder.onnx"
            export_vision_encoder_onnx(model_path, onnx_path)
            build_trt_fp16_engine(onnx_path, engine_path, args.max_batch, args.workspace_gb)

    print()
    print("[done] TRT vision encoder engine is ready.")
    print(f"  Engine path: {engine_path}")
    print(
        "  The VLM layer will auto-discover this engine because it is named\n"
        "  'vision_encoder_fp16.trt' inside the model directory.\n"
        "  Alternatively set  config_vlm_trt_vision_engine: <path>  in the config YAML."
    )


if __name__ == "__main__":
    main()
