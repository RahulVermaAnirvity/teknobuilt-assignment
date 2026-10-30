"""Settings for the P6 classifier. Secrets come from the project's .env (git-ignored)."""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    for env in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                key, sep, value = line.strip().partition("=")
                if sep and key and not key.startswith("#"):
                    os.environ.setdefault(key.strip(), value.strip().strip('"\''))
            return


_load_dotenv()
os.environ.setdefault("USE_TF", "0")  # sentence-transformers: torch only
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# --- Retrieval tier -------------------------------------------------------------
EMBED_MODEL = os.getenv("P6_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RRF_K = 60
# Fusion weights for reciprocal-rank fusion (semantic + BM25). Measured with evaluate.py on
# the sample (328 rule-labelled activities, same-name twins hidden):
#     semantic only  top-1 76.2%  top-5 93.9%   <- best
#     hybrid 1:0.3   top-1 75.0%  top-5 93.0%
#     hybrid 1:1     top-1 73.5%  top-5 91.8%
#     BM25 only      top-1 65.9%  top-5 89.6%
# BM25 adds nothing here because the exact keywords it is good at are already consumed by
# the rule tier before retrieval runs. The hybrid path stays in place; re-run evaluate.py
# on a human-labelled gold set before raising P6_W_BM25.
W_SEMANTIC = float(os.getenv("P6_W_SEMANTIC", "1.0"))
W_BM25 = float(os.getenv("P6_W_BM25", "0.0"))
TOP_K = 5
# Retrieval may label an activity on its own only if the top AUTO_NEIGHBOURS neighbours
# all share the label AND the client prefix, and the best cosine similarity is >= this.
# Calibrated with evaluate.py: ~98% precision at 0.70-0.85 (small samples: 43-89 items).
AUTO_ACCEPT_SIM = float(os.getenv("P6_AUTO_ACCEPT_SIM", "0.80"))
AUTO_NEIGHBOURS = 3

# --- LLM tier (RAG) -------------------------------------------------------------
# auto: Gemini if GEMINI_API_KEY is set, else Claude Batch API if ANTHROPIC_API_KEY, else mock
LLM_PROVIDER = os.getenv("P6_LLM_PROVIDER", "auto").lower()
GEMINI_MODELS = [m.strip() for m in os.getenv(
    "P6_GEMINI_MODELS", "gemini-flash-lite-latest,gemini-flash-latest,gemini-3.8-flash").split(",") if m.strip()]
# Second opinion for items the first model was unsure about (tried in order)
GEMINI_ESCALATION_MODELS = [m.strip() for m in os.getenv(
    "P6_GEMINI_ESCALATION_MODELS", "gemini-flash-latest,gemini-3.8-flash").split(",") if m.strip()]
LLM_BATCH = 30            # activities per request
LLM_TIMEOUT_S = 60
