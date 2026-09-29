#!/usr/bin/env sh
# Font Detector setup (first time only; needs internet)
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3.10}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m font_detector download-tessdata
if ! .venv/bin/python -m font_detector check-ocr; then
    echo
    echo "Tesseract OCR was not found. Detection of text inside images is disabled."
    echo "Install it (e.g. 'sudo apt install tesseract-ocr' or 'brew install tesseract')"
    echo "or set FONT_DETECTOR_TESSERACT to the tesseract executable."
fi
echo "Setup complete. Run ./start.sh to launch Font Detector."
