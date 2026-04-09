@echo off
setlocal
cd /d "%~dp0"

echo.
echo Running Qwen3.5-0.8B benchmark...
echo Default cases: mid-size text request + image request using truckimage.png
echo.
python benchmark.py --start-service %*
