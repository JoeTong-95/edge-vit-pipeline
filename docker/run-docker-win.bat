@echo off
REM Mount the full repo at /app (not only docker/). Works when run from docker/ or repo root.
cd /d "%~dp0.."
docker run -it --gpus all -v "%cd%":/app vision-dev:latest bash
