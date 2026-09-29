@echo off
rem Font Detector setup
rem   - If package\ contains wheels, Python packages are installed from it without internet.
rem   - Otherwise they are downloaded from PyPI (needs internet).
setlocal
cd /d "%~dp0"

set "PY=python"
py -3.10 -c "" >nul 2>nul && set "PY=py -3.10"

set "MODE=online"
set "PIPOPTS="
if exist "package\*.whl" (
    set "MODE=offline"
    set "PIPOPTS=--no-index --find-links package"
)
echo Install mode: %MODE%

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment with: %PY%
    %PY% -m venv .venv || goto :error
)

if "%MODE%"=="offline" (
    ".venv\Scripts\python.exe" -c "import sys, struct; sys.exit(0 if sys.version_info[:2] == (3, 10) and struct.calcsize('P') == 8 else 1)" || goto :wrong_python
)

echo [2/4] Installing packages (%MODE%)...
".venv\Scripts\python.exe" -m pip install %PIPOPTS% --upgrade pip || goto :error
".venv\Scripts\python.exe" -m pip install %PIPOPTS% -r requirements.txt || goto :error
".venv\Scripts\python.exe" -m pip install %PIPOPTS% --no-deps -e . || goto :error

echo [3/4] Tesseract language data (jpn, eng)...
if exist "tessdata\jpn.traineddata" if exist "tessdata\eng.traineddata" goto :tessdata_ok
if "%MODE%"=="offline" goto :tessdata_missing
".venv\Scripts\python.exe" -m font_detector download-tessdata && goto :tessdata_ok

:tessdata_missing
echo     jpn.traineddata / eng.traineddata are not in tessdata\.
echo     Copy them there to enable detection of text inside images
echo     (not needed if your Tesseract installation already includes Japanese).
goto :check_ocr

:tessdata_ok
echo     OK

:check_ocr
echo [4/4] Checking Tesseract OCR...
".venv\Scripts\python.exe" -m font_detector check-ocr && goto :done
if "%MODE%"=="offline" goto :no_tesseract

where winget >nul 2>nul || goto :no_tesseract
echo.
set "ANS="
set /p "ANS=Tesseract OCR is not installed. Install it now with winget? [y/N]: "
if /i not "%ANS%"=="y" goto :no_tesseract
winget install -e --id UB-Mannheim.TesseractOCR --accept-source-agreements --accept-package-agreements
".venv\Scripts\python.exe" -m font_detector check-ocr && goto :done

:no_tesseract
echo.
echo Tesseract OCR is not ready. Detection of text inside images is disabled,
echo but everything else works. To enable it, install Tesseract from
echo     https://github.com/UB-Mannheim/tesseract/wiki
echo and run setup.bat again (or set FONT_DETECTOR_TESSERACT to tesseract.exe).

:done
echo.
echo Setup complete. Run start.bat to launch Font Detector.
pause
exit /b 0

:wrong_python
echo.
echo package\ contains packages for 64-bit Python 3.10 only.
echo Install 64-bit Python 3.10, delete the .venv folder, and run setup.bat again.
pause
exit /b 1

:error
echo.
echo Setup failed. See the messages above.
pause
exit /b 1
