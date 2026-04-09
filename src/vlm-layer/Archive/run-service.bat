@echo off
setlocal
cd /d "%~dp0"
python service.py %*
