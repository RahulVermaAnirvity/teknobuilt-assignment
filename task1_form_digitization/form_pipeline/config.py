"""Tunable thresholds for the form pipeline.

All pixel values assume pages are rendered at RENDER_DPI. Keeping them in one
place makes it possible to recalibrate against a labelled validation set
without touching extraction logic.
"""
import os
import shutil
from pathlib import Path


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from the project's .env (secrets stay out of source control).
    Real environment variables win over the file."""
    for env in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                key, sep, value = line.strip().partition("=")
                if sep and key and not key.startswith("#"):
                    os.environ.setdefault(key.strip(), value.strip().strip('"\''))
            return


_load_dotenv()

RENDER_DPI = 200
PAGE_TIME_LIMIT_S = int(os.getenv("PAGE_TIME_LIMIT_S", "900"))  # hard cap per page (Celery mode)

# --- Page quality gate -------------------------------------------------------
BLANK_INK_RATIO = 0.004          # < 0.4% dark pixels -> page is effectively blank
HEAVY_NOISE_INK_RATIO = 0.35     # > 35% dark pixels -> mud / photocopy toner / black page
BLUR_LAPLACIAN_VAR = 400.0       # clean 200-dpi scans of this form score ~1500-1700; a 5px
                                 # gaussian blur drops to ~180, a 2x down/up-sampled fax to ~400
MIN_GRID_H_LINES = 8             # a real form has many table rules
MIN_GRID_V_LINES = 6
MIN_ANCHORS_FOUND = 4            # printed labels ("Observer", "Shift", ...) we must locate
MAX_DESKEW_DEG = 20.0
VOID_MARKERS = ("FAILED TEST", "VOID", "TEST FORM", "SPOILED", "CANCELLED", "DO NOT USE")

# --- Checkbox detection -----------------------------------------------------
INK_THRESHOLD = 110              # grey level below which a pixel counts as pen ink (mud is lighter)
CHECKBOX_MARGIN = 0.12           # fraction of the cell trimmed on each side to ignore borders
CHECKBOX_MARKED_MIN = 0.006      # absolute ink fraction needed to call a box "marked"
                                 # (a thin pen "x" covers ~1.5-3% of a box; empty boxes read ~0)
CHECKBOX_RELATIVE = 2.5          # winner must exceed the runner-up by this factor
CHECKBOX_FILLED = 0.45           # box nearly solid -> scribbled out / blacked, ambiguous

# --- Text recognition --------------------------------------------------------
HANDWRITING_MODEL = os.getenv("HANDWRITING_MODEL", "microsoft/trocr-small-handwritten")
FIELD_ACCEPT_CONF = 0.80         # below this a handwritten field is flagged for review / VLM
PRINTED_ACCEPT_CONF = 60         # tesseract word confidence (0-100)


def tesseract_cmd() -> str | None:
    """Locate the tesseract binary (env override, PATH, then the Windows default)."""
    candidates = [
        os.getenv("TESSERACT_CMD"),
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]
    return next((c for c in candidates if c and os.path.exists(c)), None)


# --- Form registry -------------------------------------------------------------
# Printed checklist variants seen so far. A new variant is flagged "unknown_form_type"
# until it is added here (one line), so new layouts never get silently mis-read.
KNOWN_FORM_TITLES = [
    "Breaking Containment - Life Saving Action",
    "O&I Confined Space - Life Saving Action",
    "Critical Devices - Life Saving Action",
]
FORM_TITLE_MATCH = 0.85
EXPECTATION_QUESTIONS = [  # fixed printed block at the foot of every form (see template)
    "Has positive feedback provided?",
    "Has the job been stopped in the event of an imminent hazard?",
    "Has an investigation been conducted?",
    "Have corrective actions been taken?",
]
ROW_ALIGN_MIN = 0.7  # answer box must overlap >= 70% with its question row

# --- Optional VLM escalation (see vlm.py) ------------------------------------
# VLM_PROVIDER: "auto" (Gemini if GEMINI_API_KEY is set, else Claude if ANTHROPIC_API_KEY), "gemini", "claude"
VLM_PROVIDER = os.getenv("VLM_PROVIDER", "auto").lower()
VLM_MODEL = os.getenv("VLM_MODEL", "claude-opus-5-5")
# Tried in order; the next model is used when one is overloaded / timing out (common on the free tier).
GEMINI_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-flash-lite-latest,gemini-flash-latest,gemini-3.8-flash").split(",") if m.strip()]
VLM_TIMEOUT_S = int(os.getenv("VLM_TIMEOUT_S", "45"))
