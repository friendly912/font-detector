@echo off
rem Font Detector setup (first time only; needs internet)
setlocal
cd /d "%~dp0"

set "PY=python"
py -3.10 -c "" >nul 2>nul && set "PY=py -3.10"

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment with: %PY%
    %PY% -m venv .venv || goto :error
)

echo [2/4] Installing packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
".venv\Scripts\python.exe" -m pip install --no-deps -e . || goto :error

echo [3/4] Downloading Tesseract language data (jpn, eng) into tessdata\ ...
".venv\Scripts\python.exe" -m font_detector download-tessdata || goto :error

echo [4/4] Checking Tesseract OCR...
".venv\Scripts\python.exe" -m font_detector check-ocr && goto :done

where winget >nul 2>nul || goto :no_tesseract
echo.
set "ANS="
set /p "ANS=Tesseract OCR is not installed. Install it now with winget? [y/N]: "
if /i not "%ANS%"=="y" goto :no_tesseract
winget install -e --id UB-Mannheim.TesseractOCR --accept-source-agreements --accept-package-agreements
".venv\Scripts\python.exe" -m font_detector check-ocr && goto :done

:no_tesseract
echo.
echo Tesseract OCR was not found. Detection of text inside images is disabled,
echo but everything else works. To enable it, install Tesseract from
echo     https://github.com/UB-Mannheim/tesseract/wiki
echo and run setup.bat again (or set FONT_DETECTOR_TESSERACT to tesseract.exe).

:done
echo.
echo Setup complete. Run start.bat to launch Font Detector.
pause
exit /b 0

:error
echo.
echo Setup failed. See the messages above.
pause
exit /b 1
