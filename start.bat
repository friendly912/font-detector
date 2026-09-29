@echo off
rem Launch Font Detector (runs fully offline)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Please run setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m font_detector serve --open %*
pause
