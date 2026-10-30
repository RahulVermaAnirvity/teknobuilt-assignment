"""Verify the environment and pre-download the open-source models.

    python scripts/check_setup.py          (run inside the activated .venv)

Checks: Python version, every pinned package imports, the Tesseract binary, optional API
keys (presence only - values are never printed), optional Redis, then downloads the TrOCR
handwriting model and the MiniLM embedding model so the first real run is fast.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "task1_form_digitization"), str(ROOT / "task2_p6_classification")]

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"
problems = 0


def line(tag, msg):
    global problems
    problems += tag == FAIL
    print(f"{tag} {msg}")


v = sys.version_info
line(OK if (3, 10) <= v[:2] <= (3, 12) else WARN,
     f"Python {v.major}.{v.minor}.{v.micro} (tested on 3.11; 3.10-3.12 supported)")
if sys.prefix == sys.base_prefix:
    line(WARN, "not running inside a virtual environment - activate .venv first to avoid conflicts")

for mod in ("pymupdf", "cv2", "numpy", "pytesseract", "pydantic", "PIL", "torch", "transformers",
            "celery", "redis", "sqlalchemy", "pandas", "sklearn", "sentence_transformers", "rank_bm25",
            "google.genai", "anthropic", "streamlit", "pyarrow"):
    try:
        m = importlib.import_module(mod)
        line(OK, f"{mod} {getattr(m, '__version__', '')}")
    except Exception as exc:  # noqa: BLE001
        line(FAIL, f"{mod}: {exc}  -> pip install -r requirements.txt")

try:
    from form_pipeline import config as C  # also loads .env
    from form_pipeline.ocr import tesseract_available
    cmd = C.tesseract_cmd()
    if tesseract_available():
        import pytesseract
        line(OK, f"Tesseract {pytesseract.get_tesseract_version()} at {cmd}")
    else:
        line(FAIL, "Tesseract binary not found. Windows: winget install UB-Mannheim.TesseractOCR | "
                   "macOS: brew install tesseract | Ubuntu: sudo apt install tesseract-ocr  "
                   "(or set TESSERACT_CMD in .env)")
except Exception as exc:  # noqa: BLE001
    line(FAIL, f"Task 1 package import failed: {exc}")

line(OK if os.getenv("GEMINI_API_KEY") else WARN,
     "GEMINI_API_KEY " + ("set" if os.getenv("GEMINI_API_KEY") else
                          "not set - optional: AI escalation (Task 1) and RAG step (Task 2) stay off"))
broker = os.getenv("CELERY_BROKER_URL")
if broker:
    try:
        import redis
        redis.Redis.from_url(broker, socket_connect_timeout=2).ping()
        line(OK, f"Redis reachable at {broker}")
    except Exception as exc:  # noqa: BLE001
        line(WARN, f"CELERY_BROKER_URL={broker} but Redis is not reachable ({exc}) - "
                   "run `docker compose up -d redis` or remove the line from .env")
else:
    line(OK, "Celery broker: zero-setup local filesystem broker (set CELERY_BROKER_URL for Redis)")

for name in ("Sample_input.pdf", "P6_Schedule_Sample.csv"):
    line(OK if (ROOT / name).exists() else WARN,
         f"{name} {'found' if (ROOT / name).exists() else 'missing - the app still works with uploads'}")

if "--skip-models" not in sys.argv:
    print("\nDownloading / verifying models (first time: ~350 MB)…")
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        name = os.getenv("HANDWRITING_MODEL", "microsoft/trocr-small-handwritten")
        TrOCRProcessor.from_pretrained(name)
        VisionEncoderDecoderModel.from_pretrained(name)
        line(OK, f"handwriting model {name}")
        from sentence_transformers import SentenceTransformer
        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        line(OK, "embedding model all-MiniLM-L6-v2")
    except Exception as exc:  # noqa: BLE001
        line(FAIL, f"model download failed: {exc}")

print("\nAll good - see README 'Run' section." if not problems else f"\n{problems} problem(s) above need fixing.")
sys.exit(1 if problems else 0)
