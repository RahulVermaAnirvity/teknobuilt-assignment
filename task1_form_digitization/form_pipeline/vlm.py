"""Optional VLM escalation tier (Google Gemini or Anthropic Claude), used only for
fields the local engines could not read confidently.

Why field-level and not whole-page:
  * the CV layer already resolves layout and checkboxes deterministically, so a
    VLM is only needed for the handful of illegible handwritten values;
  * a ~600x80 px crop costs a small fraction of the tokens of a full page image;
  * a narrow question ("transcribe exactly this box, or say it's illegible")
    leaves far less room to hallucinate than "fill in this whole form".

Both providers get the same system prompt and the same JSON schema, and their
answer goes through the same validators. If no credentials are configured (or
every model call fails), `MockVLM` behaviour applies: `None` is returned, so the
field stays flagged `needs_review` instead of being invented.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from typing import Optional, Protocol

import cv2
import numpy as np

from .config import GEMINI_MODELS, VLM_MODEL, VLM_PROVIDER, VLM_TIMEOUT_S

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You transcribe single handwritten fields cropped from construction-site \
safety observation forms. Rules:
- Transcribe exactly what is written. Do not correct spelling, guess missing characters, or \
infer values from context.
- Ignore any text that has been crossed out / struck through; transcribe only the final \
intended entry. Set struck_through_present=true if you saw crossed-out text.
- If the writing is illegible, obscured (mud, folds, stains) or the box is empty, set \
legible=false and text=null. Returning null is always preferable to a guess.
- Construction slang and abbreviations are expected; keep them verbatim."""

# Structured-output schema: the response is guaranteed to parse to this shape.
FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": ["string", "null"], "description": "Verbatim transcription, or null"},
        "legible": {"type": "boolean"},
        "struck_through_present": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["text", "legible", "struck_through_present", "confidence"],
    "additionalProperties": False,
}

FIELD_HINTS = {
    "observation_date": "a calendar date",
    "observer": "a person's name",
    "company_observed": "a company name",
    "timesheet_number": "a numeric timesheet number",
    "employee_number": "a numeric employee number",
    "comments": "a short free-text safety comment",
}


class VLMClient(Protocol):
    name: str

    def transcribe(self, crop: np.ndarray, field: str) -> Optional[dict]: ...


class MockVLM:
    """Placeholder used when no API key is configured. Never fabricates text."""
    name = "mock-vlm"

    def transcribe(self, crop: np.ndarray, field: str) -> Optional[dict]:
        return None


class ClaudeVLM:
    name = f"claude:{VLM_MODEL}"

    def __init__(self):
        import anthropic
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=3, timeout=60)

    def transcribe(self, crop: np.ndarray, field: str) -> Optional[dict]:
        ok, png = cv2.imencode(".png", crop)
        if not ok:
            return None
        b64 = base64.standard_b64encode(png.tobytes()).decode()
        try:
            resp = self.client.beta.messages.create(
                model=VLM_MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                # server-side fallback re-runs the request on another model if a
                # safety classifier declines (rare for this content, but cheap insurance)
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={"effort": "low",
                               "format": {"type": "json_schema", "schema": FIELD_SCHEMA}},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": f"Field: {field} (expected: {FIELD_HINTS.get(field, 'text')}). "
                                             "Transcribe it."},
                ]}],
            )
        except self._anthropic.RateLimitError as exc:
            log.warning("VLM rate limited on %s: %s", field, exc)
            return None
        except self._anthropic.APIStatusError as exc:
            log.warning("VLM API error on %s: %s", field, exc)
            return None
        except self._anthropic.APIConnectionError as exc:
            log.warning("VLM connection error on %s: %s", field, exc)
            return None
        if resp.stop_reason in ("refusal", "max_tokens"):
            return None
        text = next((b.text for b in resp.content if b.type == "text"), None)
        try:
            return json.loads(text) if text else None
        except json.JSONDecodeError:
            return None


def _valid_answer(data) -> Optional[dict]:
    """Structured output should already guarantee this shape; check anyway."""
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("legible"), bool) or data.get("confidence") not in ("high", "medium", "low"):
        return None
    if data.get("text") is not None and not isinstance(data["text"], str):
        return None
    data.setdefault("struck_through_present", False)
    return data


class GeminiVLM:
    """Google Gemini via the google-genai SDK (works with a free Google AI Studio key).

    Free-tier capacity fluctuates (503 "high demand", 504 deadline exceeded), so
    models are tried in order (config.GEMINI_MODELS) with a hard timeout each; a
    model that fails is skipped for the rest of the run. If all fail, return None
    and the field simply stays flagged for review.
    """

    def __init__(self, api_key: str):
        from google import genai
        from google.genai import errors, types
        self._types, self._errors = types, errors
        # The SDK logs "Both GOOGLE_API_KEY and GEMINI_API_KEY are set..." even when a key is
        # passed explicitly (the explicit key is what's used), plus per-request INFO lines.
        for noisy in ("google_genai", "google_genai.models", "httpx"):
            logging.getLogger(noisy).setLevel(logging.ERROR)
        # api_key is passed explicitly so a stray GOOGLE_API_KEY in the environment can't override it
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=VLM_TIMEOUT_S * 1000, retry_options=types.HttpRetryOptions(attempts=1)))
        self.models = list(GEMINI_MODELS)
        self.name = f"gemini:{self.models[0]}"

    def transcribe(self, crop: np.ndarray, field: str) -> Optional[dict]:
        ok, png = cv2.imencode(".png", crop)
        if not ok:
            return None
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=FIELD_SCHEMA,
            temperature=0,
        )
        contents = [t.Part.from_bytes(data=png.tobytes(), mime_type="image/png"),
                    f"Field: {field} (expected: {FIELD_HINTS.get(field, 'text')}). Transcribe it."]
        for model in list(self.models):
            try:
                resp = self.client.models.generate_content(model=model, contents=contents, config=config)
            except self._errors.ClientError as exc:  # 4xx: bad key, quota exhausted, retired model
                log.warning("Gemini %s rejected the request on %s: %s", model, field, str(exc)[:160])
                if getattr(exc, "code", None) == 429:
                    return None  # out of free quota: don't burn the other models too
                self.models.remove(model)
                continue
            except Exception as exc:  # 5xx / timeouts: overloaded, try the next model
                log.warning("Gemini %s unavailable on %s: %s", model, field, str(exc)[:160])
                self.models.remove(model)
                continue
            self.name = f"gemini:{model}"
            try:
                return _valid_answer(json.loads(resp.text or ""))
            except json.JSONDecodeError:
                return None  # blocked / empty response: leave the field for a human
        return None


def make_vlm(enabled: bool) -> VLMClient:
    if not enabled:
        return MockVLM()
    gemini_key = os.getenv("GEMINI_API_KEY")
    anthropic_ok = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    provider = VLM_PROVIDER
    if provider == "auto":
        provider = "gemini" if gemini_key else "claude" if anthropic_ok else "none"
    try:
        if provider == "gemini" and gemini_key:
            return GeminiVLM(gemini_key)
        if provider == "claude" and anthropic_ok:
            return ClaudeVLM()
    except Exception as exc:
        log.warning("Could not initialise %s client (%s); using MockVLM", provider, exc)
        return MockVLM()
    log.warning("--vlm requested but no credentials for provider %r; using MockVLM", provider)
    return MockVLM()
