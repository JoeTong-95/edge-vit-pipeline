# edge-vision-pipeline
Enabling VLM applications on edge devices such as Jetson Orin Nano.

# Docker update and usage

All Docker-related files live under the `docker/` folder:

- `docker/Dockerfile` / `docker/Dockerfile.dev` — dev machine images
- `docker/requirements.txt` / `docker/requirements.dev.txt` — dev machine deps
- `docker/setup-native-jetson.sh` — **Jetson native install** (recommended for Jetson)

> **Jetson note**: Docker-based GPU inference on Jetson Orin is blocked by a
> PyTorch NvMap allocator incompatibility (see `JETSON_OPTIMIZATION.md`).
> Use `docker/setup-native-jetson.sh` for native installation on Jetson instead.

## Docker environment setup

This project uses Docker for dev-machine development:

- `docker/Dockerfile` or `docker/Dockerfile.dev` for the main development machine
- For Jetson, use `docker/setup-native-jetson.sh` (native, no Docker)

Current convention:

- `docker/Dockerfile` is the default dev-machine Dockerfile
- `docker/Dockerfile.dev` is the explicit dev-machine Dockerfile
- `docker/requirements.dev.txt` is the dev-machine Python dependency set
- `docker/requirements.txt` mirrors the dev dependency set for compatibility and quick inspection

Helper scripts (`build-docker-*`, `run-docker-*`) live in `docker/`. They **change to the repository root** (parent of `docker/`) before running `docker build` or `docker run`, so paths like `-f docker/Dockerfile.dev` and the volume mount for `/app` stay correct whether you run them from the repo root, from `docker/`, or by double-clicking on Windows.

### Build on the dev machine

Run this from the project root directory:

```bash
docker build --pull -t vision-dev:latest -f docker/Dockerfile docker
```

Equivalent explicit command:

```bash
docker build --pull -t vision-dev:latest -f docker/Dockerfile.dev docker
```

Helper scripts:

Linux:

```bash
./docker/build-docker-dev-machine
```

Windows:

```powershell
.\docker\build-docker-dev-machine.bat
```

### Install on Jetson (native, no Docker)

```bash
# Installs torch 2.8.0 (CUDA 12.6 wheel) + all pipeline deps
bash docker/setup-native-jetson.sh

# Run benchmark
BENCH_CONFIG_YAML=src/configuration-layer/config.jetson.yaml python3 benchmark.py
```

### Run the dev-machine container

Linux:

```bash
docker run -it --gpus all -v ${PWD}:/app vision-dev:latest bash
```

Helper script:

```bash
./docker/run-docker-linux
```

macOS:

```bash
docker run -it -v "$(pwd)":/app vision-dev:latest bash
```

Helper script:

```bash
./docker/run-docker-mac
```

Windows:

```bat
docker run -it --gpus all -v %cd%:/app vision-dev:latest bash
```

Helper script:

```powershell
.\docker\run-docker-win.bat
```

### Hugging Face CLI

Inside either container, the Hugging Face CLI should be available:

```bash
huggingface-cli --help
```

You can log in for model downloads with:

```bash
huggingface-cli login
```

### Typical workflow

1. Build `vision-dev:latest` on the dev machine.
2. Edit and iterate locally.
3. On Jetson, run `bash docker/setup-native-jetson.sh` (native install).
4. Validate YOLO, VLM, GPU, and deployment behavior on the Jetson using `config.jetson.yaml`.

### Visualization and VLM demos on the host

Config-driven helpers (YOLO-only, tracking, ROI, cropper cache, and the end-to-end `src/vlm-layer/visualize_vlm.py` path) read `src/configuration-layer/config.yaml` and are listed in `pipeline/README.md`. Mount this repo into the container the same way you mount it for development so those scripts see `config.yaml`, video paths, and local model weights.

The bundled Qwen3.5 VLM weights require **`transformers` 5.x** (`model_type` `qwen3_5`). The `docker/requirements*.txt` files pin `transformers>=5.0.0,<6.0.0`; upgrade the image or your venv if you still have 4.x installed.

### Notes

- The repository is mounted from the host, so code changes do not require rebuilding the image.
- Rebuild when the relevant Dockerfile or requirements file changes.
- `docker/Dockerfile.jetson` is for ARM64 Jetson systems and will not build normally on an x86 Docker Desktop environment.
- `run-docker-jetson` requests GPU access with `--runtime=nvidia`, which is the Jetson-compatible runtime setting.
- `run-docker-linux` and `run-docker-win.bat` request GPU passthrough with `--gpus all`.
- `run-docker-mac` intentionally runs without GPU flags.
- On Linux, run `chmod +x docker/build-docker-dev-machine docker/run-docker-linux docker/run-docker-mac` once after cloning if execute bits are missing.
- The dev image installs `opencv-python` from pip.
- The Jetson image installs `python3-opencv` from apt and avoids the pip OpenCV wheel to reduce Jetson-specific `cv2` conflicts.
