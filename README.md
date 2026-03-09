# edge-vision-pipeline
Enabling VLM applications on edge devices such as Jetson Orin Nano

# Docker Update and Usage 

## Docker Environment Setup

This project uses Docker to provide a consistent development environment.

### Build the Docker image

From the project root directory:

docker build -t vision-dev -f docker/Dockerfile .

This builds the development image `vision-dev`.

### Run the container

docker run -it --gpus all -v ${PWD}:/app vision-dev bash

Explanation:

- `-it` → interactive terminal
- `--gpus all` → enables GPU access inside the container
- `-v ${PWD}:/app` → mounts the current project directory into the container

Inside the container, the project will appear at:

/app

This means any edits made on your host machine are immediately visible inside the container.

### Verify GPU availability

Inside the container, run:

python -c "import torch; print(torch.cuda.is_available())"

Expected output:

True

- if using Nvidia Graphics, check with
```nvidia-smi```

- check docker CUDA version with 
```python -c "import torch; print(torch.version.cuda)"```

### Typical development workflow

1. Start the container:

docker run -it --gpus all -v ${PWD}:/app vision-dev bash

2. Work inside `/app`:

cd /app

3. Run scripts or experiments normally.

### Notes

- The repository is mounted from the host, so code changes do not require rebuilding the image.
- Rebuild the Docker image only if the Dockerfile or dependencies change.