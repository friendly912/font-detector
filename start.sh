#!/usr/bin/env sh
# Launch Font Detector (runs fully offline)
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Please run ./setup.sh first."; exit 1; }
exec .venv/bin/python -m font_detector serve --open "$@"
