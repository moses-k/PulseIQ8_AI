@echo off
REM Runs PulseIQ8_AI on Windows (port 8091 by default). Double-click to run; close the window to stop.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
pause
