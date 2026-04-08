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
    args = parser.parse_args()

    layer = _load_layer_module()
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


if __name__ == '__main__':
    main()
