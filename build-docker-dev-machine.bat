@echo off
docker build --pull -t vision-dev:latest -f docker/Dockerfile.dev docker
