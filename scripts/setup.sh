#!/usr/bin/env bash
# macOS / Linux setup: isolated virtual environment in .venv + pinned dependencies.
#   bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=""
for candidate in python3.11 python3.12 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && \
     "$candidate" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)'; then
    PY="$candidate"; break
  fi
done
[ -n "$PY" ] || { echo "Python 3.10-3.12 not found - install Python 3.11"; exit 1; }
echo "Using $PY"

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example - add your GEMINI_API_KEY to enable the AI steps (optional)."
fi

if ! command -v tesseract >/dev/null 2>&1; then
  echo "WARNING: Tesseract OCR not found. Install it with:"
  echo "  macOS:  brew install tesseract"
  echo "  Ubuntu: sudo apt-get install -y tesseract-ocr"
fi

.venv/bin/python scripts/check_setup.py
echo
echo "Activate the environment with:  source .venv/bin/activate"
