# edge-vision-pipeline
Enabling VLM applications on edge devices such as Jetson Orin Nano.

# Docker update and usage

All Docker-related files now live under the `docker/` folder:

- `docker/Dockerfile`
- `docker/requirements.txt`
- `docker/run-lix-docker-lateset-gpu-current_dir.bash`
- `docker/run-win--docker-lateset-gpu-current_dir.bat`

## Docker environment setup

This project uses Docker to provide a consistent development environment.

### Build the Docker image

Run this from the project root directory:

```bash
docker build -t vision-dev:latest -f docker/Dockerfile .
```

This builds the development image `vision-dev:latest`.

### Run the container

From the project root directory:

Linux/macOS:

```bash
bash docker/run-lix-docker-lateset-gpu-current_dir.bash
```

Windows:

```bat
docker\run-win--docker-lateset-gpu-current_dir.bat
```

Equivalent direct commands:

Linux/macOS:

```bash
docker run -it --gpus all -v ${PWD}:/app vision-dev:latest bash
```

Windows:

```bat
docker run -it --gpus all -v %cd%:/app vision-dev:latest bash
```

Explanation:

- `-it` opens an interactive terminal.
- `--gpus all` enables GPU access inside the container.
- `-v ${PWD}:/app` or `-v %cd%:/app` mounts the current project directory into the container.

Inside the container, the project appears at:

```text
/app
```

This means edits made on the host machine are immediately visible inside the container.

### Verify GPU availability

Inside the container, run:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```text
True
```

If using NVIDIA graphics, you can also check:

```bash
nvidia-smi
```

Check the CUDA version seen by PyTorch with:

```bash
python -c "import torch; print(torch.version.cuda)"
```

### Typical development workflow

1. Build the image from the project root.
2. Start the container using one of the `docker/` run scripts.
3. Work inside `/app`.
4. Run scripts or experiments normally.

### Notes

- The repository is mounted from the host, so code changes do not require rebuilding the image.
- Rebuild the image only if `docker/Dockerfile` or `docker/requirements.txt` changes.
- Run the helper scripts from the project root so the current directory is mounted correctly.
