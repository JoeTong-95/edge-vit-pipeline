from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image


LAYER_DIR = Path(__file__).resolve().parent
LAYER_PATH = LAYER_DIR / 'layer.py'
DEFAULT_IMAGE_PATH = LAYER_DIR / 'truckimage.png'


def _load_layer_module():
    module_name = 'vlm_layer_runtime'
    spec = importlib.util.spec_from_file_location(module_name, LAYER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load layer module from {LAYER_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a smoke test for the VLM layer.')
    parser.add_argument('--image', type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument('--track-id', default='burner-track-001')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--query-type', default='vehicle_semantics_v1')
    parser.add_argument('--disabled', action='store_true')
    parser.add_argument('--sample-only', action='store_true', help='Print sample VLM output JSON strings without loading the model.')
    parser.add_argument('--output-dir', type=Path, default=None, help='Directory for saved VLM debug images.')
    args = parser.parse_args()

    layer = _load_layer_module()
    output_dir = args.output_dir if args.output_dir is not None else layer.DEFAULT_VLM_DEBUG_OUTPUT_DIR
    if args.sample_only:
        print('Sample VLM output JSON strings:')
        for sample_json in layer.build_sample_vlm_output_json_strings():
            print(sample_json)
            print()
        sample_image = Image.open(DEFAULT_IMAGE_PATH).convert('RGB')
        saved_paths = layer.save_sample_vlm_output_debug_images(
            sample_image=sample_image,
            output_dir=output_dir,
        )
        print('Saved sample debug images:')
        for saved_path in saved_paths:
            print(saved_path)
        return

    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f'Smoke-test image not found: {image_path}')

    config = layer.VLMConfig(
        config_vlm_enabled=not args.disabled,
        config_vlm_model=str(layer.DEFAULT_VLM_MODEL_PATH),
        config_device=args.device,
    )

    print('Initializing VLM layer...')
    runtime_state = layer.initialize_vlm_layer(config)
    print(json.dumps({
        'config_vlm_enabled': runtime_state.config_vlm_enabled,
        'vlm_runtime_device': runtime_state.vlm_runtime_device,
        'vlm_runtime_dtype': runtime_state.vlm_runtime_dtype,
        'vlm_runtime_model_id': runtime_state.vlm_runtime_model_id,
    }, indent=2))

    if args.disabled:
        print('Layer disabled smoke test complete.')
        return

    crop_image = Image.open(image_path).convert('RGB')
    burner_package = layer.VLMFrameCropperLayerPackage(
        vlm_frame_cropper_layer_track_id=args.track_id,
        vlm_frame_cropper_layer_image=crop_image,
        vlm_frame_cropper_layer_bbox=None,
    )

    print('Running inference with burner cropper package...')
    raw_result = layer.run_vlm_inference(
        vlm_runtime_state=runtime_state,
        vlm_frame_cropper_layer_package=burner_package,
        vlm_layer_query_type=args.query_type,
    )
    normalized_result = layer.normalize_vlm_result(raw_result)
    layer_package = layer.build_vlm_layer_package(raw_result)
    ack_package = layer.build_vlm_ack_package_from_result(raw_result)

    print('\nRaw result:')
    print(json.dumps({
        'vlm_layer_track_id': raw_result.vlm_layer_track_id,
        'vlm_layer_query_type': raw_result.vlm_layer_query_type,
        'vlm_layer_model_id': raw_result.vlm_layer_model_id,
        'vlm_layer_raw_text': raw_result.vlm_layer_raw_text,
    }, indent=2))

    print('\nNormalized result:')
    print(json.dumps(normalized_result, indent=2))

    print('\nLayer package:')
    print(json.dumps(layer.serialize_vlm_layer_package(layer_package), indent=2))

    print('\nAck package:')
    print(json.dumps(layer.serialize_vlm_ack_package(ack_package), indent=2))

    print('\nCombined output JSON:')
    print(layer.format_vlm_output_json(raw_result))

    saved_debug_image = layer.save_vlm_debug_image(
        vlm_frame_cropper_layer_package=burner_package,
        vlm_layer_raw_result=raw_result,
        output_dir=output_dir,
    )
    print('\nSaved debug image:')
    print(saved_debug_image)


if __name__ == '__main__':
    main()
