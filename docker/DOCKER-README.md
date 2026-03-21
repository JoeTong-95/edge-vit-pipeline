# edge-vision-pipeline
Enabling VLM applications on edge devices such as Jetson Orin Nano.

# Docker update and usage

All Docker-related files live under the `docker/` folder:

- `docker/Dockerfile`
- `docker/Dockerfile.dev`
- `docker/Dockerfile.jetson`
- `docker/requirements.txt`

## Docker environment setup

This project uses two Docker images:

- `docker/Dockerfile` or `docker/Dockerfile.dev` for the main development machine
- `docker/Dockerfile.jetson` for Jetson verification

Both Dockerfiles can live in the same folder. Choose which one to build with `-f`.

Current convention:

- `docker/Dockerfile` is the default dev-machine Dockerfile
- `docker/Dockerfile.dev` is the explicit dev-machine Dockerfile
- `docker/Dockerfile.jetson` is the Jetson-only Dockerfile

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
./build-docker-dev-machine
```

Windows:

```powershell
.\build-docker-dev-machine.bat
```

### Build on the Jetson

Run this from the project root directory on the Jetson:

```bash
docker build --pull -t vision-jetson:latest -f docker/Dockerfile.jetson docker
```

Helper script:

```bash
./build-docker-jetson
```

### Run the dev-machine container

Linux:

```bash
docker run -it --gpus all -v ${PWD}:/app vision-dev:latest bash
```

Helper script:

```bash
./run-docker-linux
```

macOS:

```bash
docker run -it -v "$(pwd)":/app vision-dev:latest bash
```

Helper script:

```bash
./run-docker-mac
```

Windows:

```bat
docker run -it --gpus all -v %cd%:/app vision-dev:latest bash
```

Helper script:

```powershell
.\run-docker-win.bat
```

### Run the Jetson container

On the Jetson:

```bash
docker run -it --gpus all -v ${PWD}:/app vision-jetson:latest bash
```

Helper script:

```bash
./run-docker-jetson
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
3. Build `vision-jetson:latest` on the Jetson.
4. Validate YOLO, VLM, GPU, and deployment behavior on the Jetson.

### Notes

- The repository is mounted from the host, so code changes do not require rebuilding the image.
- Rebuild when `docker/requirements.txt` or the relevant Dockerfile changes.
- `docker/Dockerfile.jetson` is for ARM64 Jetson systems and will not build normally on an x86 Docker Desktop environment.
- `run-docker-jetson`, `run-docker-linux`, and `run-docker-win.bat` request GPU passthrough with `--gpus all`.
- `run-docker-mac` intentionally runs without GPU flags.
- On Linux and Jetson, run `chmod +x build-docker-dev-machine build-docker-jetson run-docker-linux run-docker-mac run-docker-jetson` once after cloning if execute bits are missing.
