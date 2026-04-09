# VLM Frame Cropper Layer

This layer prepares object-level image crops for semantic reasoning.

## Public API

- `build_vlm_frame_cropper_request_package`
- `extract_vlm_object_crop`
- `build_vlm_frame_cropper_package`

## Ownership

- Builds one crop request for one tracked object.
- Resolves the source frame from the input layer package.
- Extracts and validates the crop before handing it to the VLM layer.

## Quick Start

From the project root:

```powershell
python src/vlm-frame-cropper-layer/test_vlm_frame_cropper_layer.py
```
