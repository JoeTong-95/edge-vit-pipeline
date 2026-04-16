@echo off
REM Build context is the docker/ folder (requirements*.txt live there).
REM Run from repo root or from docker/ — we always cd to repo root first.
cd /d "%~dp0.."
docker build --pull -t vision-dev:latest -f docker/Dockerfile.dev docker
