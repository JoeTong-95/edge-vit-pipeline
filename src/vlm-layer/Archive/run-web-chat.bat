@echo off
setlocal
cd /d "%~dp0"

echo.
echo Starting Qwen3.5-0.8B local web chat...
echo Visit: http://127.0.0.1:8010/
echo Keep this window open while you use the browser UI.
echo.
python service.py --host 127.0.0.1 --port 8010 %*
