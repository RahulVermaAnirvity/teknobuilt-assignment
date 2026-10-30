"""Tier 3 - retrieval-augmented LLM classification of the residual activities.

Each activity is sent with its retrieved evidence: the most similar already-labelled
activities (with their labels and similarity) and the candidate labels they vote for.
The model decides; retrieval only informs.

Constraint enforcement (layered):
  1. the output schema makes `label` an `enum` of the 41 valid "Discipline :: Sub-Discipline"
     pairs + "UNCERTAIN" - constrained decoding cannot produce anything else;
  2. "UNCERTAIN" gives the model a sanctioned way out instead of forcing a guess;
  3. every returned label is re-validated in code; missing/invalid -> retry -> review queue.

Providers: Gemini (live, free Google AI Studio key in .env) is the default; the Claude
Message Batches API path is kept for large asynchronous runs; MockRunner never invents.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from . import config as C
from .taxonomy import LABEL_SET, LABELS, TAXONOMY

log = logging.getLogger(__name__)

CLAUDE_WORKER_MODEL = os.getenv("P6_WORKER_MODEL", "claude-haiku-5-5")

_taxonomy_text = "\n".join(f"- {d}: " + "; ".join(subs) for d, subs in TAXONOMY.items())

SYSTEM_PROMPT = f"""You classify construction schedule activities (Primavera P6, data-centre \
projects) into a fixed Discipline / Sub-Discipline taxonomy.

TAXONOMY (the only permitted answers):
{_taxonomy_text}

Each input activity comes with evidence:
- `similar_examples`: previously classified activities from the same programme, most similar
  first, with their verified label and a similarity score (0-1). Use them as precedent, but
  they are only similar, not identical - check the actual wording.
- `candidate_labels`: the labels those examples vote for. The answer is often among them,
  but you may choose any taxonomy label.
- `client_prefix` (ARC, CON, ELE, MEC, SEC, TELE ...): the client's own discipline code. It is
  frequently wrong; use it only to disambiguate generic names such as "Testing" or "Cabling".
- `rule_hint`: a low-confidence keyword-rule suggestion. It may be wrong.

How to decide:
- Classify by the physical work described, not by who the client assigned it to.
- Containment / cable tray is Electrical (main tray = Cable Trays Main, drops/secondary =
  Cable Trays Drops) unless it is telecom basket tray.
- Below-ground services (foul, storm, potable water, MV/LV ducts) are Civil underground
  utilities; above-ground building services are Plumbing / Mechanical / Electrical.
- Elevated concrete (upper-level slabs, cores, walls) is Structural; ground-bearing slabs
  are Civil Slab on Grade.
- If the name is too vague to classify with reasonable confidence (e.g. an unexplained
  acronym, or "Fit-Out" with no trade), answer UNCERTAIN. Never invent a category.
- `confidence`: "high" only if you would bet on it; otherwise "medium" or "low".
- `reason`: at most 15 words.

Return exactly one result per input `key`."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "label": {"type": "string", "enum": LABELS + ["UNCERTAIN"]},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reason": {"type": "string"},
                },
                "required": ["key", "label", "confidence", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


@dataclass
class Item:
    key: str
    name: str            # one representative raw activity name
    prefix: str
    hint: Optional[str]  # low-confidence rule suggestion, if any
    examples: list[dict] = field(default_factory=list)   # [{name, client_prefix, label, similarity}]
    candidates: list[str] = field(default_factory=list)

    def payload(self) -> dict:
        return {"key": self.key, "name": self.name, "client_prefix": self.prefix, "rule_hint": self.hint,
                "similar_examples": self.examples, "candidate_labels": self.candidates}


@dataclass
class Decision:
    label: str           # a taxonomy label or "UNCERTAIN"
    confidence: str      # high | medium | low
    reason: str
    model: str


def chunks(items: list[Item], n: int = C.LLM_BATCH) -> Iterable[list[Item]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def parse_results(text: str, expected_keys: set[str]) -> tuple[dict[str, tuple[str, str, str]], set[str]]:
    """Validate a model response -> ({key: (label, confidence, reason)}, keys_to_retry)."""
    out: dict[str, tuple[str, str, str]] = {}
    try:
        rows = json.loads(text)["results"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}, set(expected_keys)
    for r in rows:
        if not isinstance(r, dict):
            continue
        k, lab = r.get("key"), r.get("label")
        conf = r.get("confidence") if r.get("confidence") in ("high", "medium", "low") else "low"
        if k in expected_keys and (lab in LABEL_SET or lab == "UNCERTAIN"):
            out[k] = (lab, conf, str(r.get("reason", ""))[:200])
    return out, expected_keys - out.keys()


# --------------------------------------------------------------------------- providers

class GeminiClassifier:
    """Synchronous Gemini calls (free AI Studio key). Models are tried in order; one that is
    overloaded / timing out is skipped for the rest of the run; quota exhaustion stops the tier."""

    def __init__(self, api_key: str, models: list[str] | None = None):
        import logging as _logging
        from google import genai
        from google.genai import errors, types
        for noisy in ("google_genai", "google_genai.models", "httpx"):
            _logging.getLogger(noisy).setLevel(_logging.ERROR)
        self._types, self._errors = types, errors
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=C.LLM_TIMEOUT_S * 1000, retry_options=types.HttpRetryOptions(attempts=1)))
        self.models = list(models or C.GEMINI_MODELS)
        self.quota_exhausted = False
        self.name = "gemini"

    def classify(self, items: list[Item]) -> dict[str, Decision]:
        if self.quota_exhausted or not items:
            return {}
        t = self._types
        config = t.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0,
                                         response_mime_type="application/json",
                                         response_json_schema=OUTPUT_SCHEMA)
        content = json.dumps([it.payload() for it in items], ensure_ascii=False)
        expected = {it.key for it in items}
        for model in list(self.models):
            resp = None
            for attempt in range(3):  # free-tier overload is usually a blip: back off and retry
                try:
                    resp = self.client.models.generate_content(model=model, contents=content, config=config)
                    break
                except self._errors.ClientError as exc:
                    log.warning("Gemini %s rejected the request: %s", model, str(exc)[:160])
                    if getattr(exc, "code", None) == 429:
                        self.quota_exhausted = True
                        return {}
                    self.models.remove(model)  # 400/404: retired model or bad request - skip for good
                    break
                except Exception as exc:  # 5xx / timeout
                    log.warning("Gemini %s unavailable (attempt %d/3): %s", model, attempt + 1, str(exc)[:120])
                    time.sleep(2 + 4 * attempt)
            if resp is None:
                if model in self.models:  # failed 3 times in a row: overloaded right now, skip it this run
                    self.models.remove(model)
                continue
            parsed, missing = parse_results(resp.text or "", expected)
            if missing:
                finish = resp.candidates[0].finish_reason if resp.candidates else "no candidates"
                log.warning("Gemini %s left %d/%d item(s) unanswered (finish=%s, %d chars returned)",
                            model, len(missing), len(expected), finish, len(resp.text or ""))
            self.name = f"gemini:{model}"
            return {k: Decision(lab, conf, reason, f"gemini:{model}") for k, (lab, conf, reason) in parsed.items()}
        return {}


class ClaudeBatchClassifier:
    """Message Batches API (50% price, asynchronous) for very large runs."""
    name = f"claude:{CLAUDE_WORKER_MODEL}"

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()

    @staticmethod
    def build_request(custom_id: str, items: list[Item]) -> dict:
        return {
            "custom_id": custom_id,
            "params": {
                "model": CLAUDE_WORKER_MODEL,
                "max_tokens": 8000,
                "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
                "messages": [{"role": "user", "content": json.dumps([it.payload() for it in items], ensure_ascii=False)}],
            },
        }

    def classify(self, items: list[Item], poll_s: int = 60) -> dict[str, Decision]:
        batches = list(chunks(items))
        reqs = [self.build_request(f"p6-{i:05d}", b) for i, b in enumerate(batches)]
        batch = self.client.messages.batches.create(requests=reqs)
        while self.client.messages.batches.retrieve(batch.id).processing_status != "ended":
            time.sleep(poll_s)
        by_id = {r["custom_id"]: {it.key for it in b} for r, b in zip(reqs, batches)}
        out: dict[str, Decision] = {}
        for res in self.client.messages.batches.results(batch.id):
            if res.result.type == "succeeded" and res.result.message.stop_reason == "end_turn":
                text = next((b.text for b in res.result.message.content if b.type == "text"), "")
                parsed, _ = parse_results(text, by_id.get(res.custom_id, set()))
                out.update({k: Decision(lab, conf, reason, self.name) for k, (lab, conf, reason) in parsed.items()})
        return out


class MockClassifier:
    """No credentials: answer nothing, so items stay in the review queue (never invented)."""
    name = "mock"

    def classify(self, items: list[Item]) -> dict[str, Decision]:
        return {}


def make_classifier(provider: str = C.LLM_PROVIDER, models: list[str] | None = None):
    gemini_key = os.getenv("GEMINI_API_KEY")
    anthropic_ok = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    if provider == "auto":
        provider = "gemini" if gemini_key else "claude" if anthropic_ok else "none"
    try:
        if provider == "gemini" and gemini_key:
            return GeminiClassifier(gemini_key, models)
        if provider == "claude" and anthropic_ok:
            return ClaudeBatchClassifier()
    except Exception as exc:
        log.warning("could not initialise %s classifier (%s); using mock", provider, exc)
    return MockClassifier()


def classify_items(items: list[Item], provider: str = C.LLM_PROVIDER, progress=None) -> tuple[dict[str, Decision], str]:
    """Worker pass over all items, then one escalation pass for UNCERTAIN / non-high answers.
    Returns ({key: Decision}, description of models used)."""
    worker = make_classifier(provider)
    if isinstance(worker, MockClassifier) or not items:
        return {}, worker.name
    decisions: dict[str, Decision] = {}
    batches = list(chunks(items))
    for i, batch in enumerate(batches, 1):
        got = worker.classify(batch)
        missing = [it for it in batch if it.key not in got]
        if missing:  # one retry for dropped / invalid answers
            got.update(worker.classify(missing))
        decisions.update(got)
        if progress:
            progress(i, len(batches))
    used = {worker.name}

    if isinstance(worker, GeminiClassifier):
        unsure = [it for it in items if it.key not in decisions
                  or decisions[it.key].label == "UNCERTAIN" or decisions[it.key].confidence != "high"]
        if unsure and not worker.quota_exhausted:
            second = GeminiClassifier(os.environ["GEMINI_API_KEY"], C.GEMINI_ESCALATION_MODELS)
            for batch in chunks(unsure):
                for k, d in second.classify(batch).items():
                    prev = decisions.get(k)
                    # take the stronger model's answer when it is confident, or when it agrees
                    if d.label != "UNCERTAIN" and (d.confidence == "high" or (prev and prev.label == d.label)):
                        if prev and prev.label == d.label and d.confidence != "high":
                            # two models reached the same label -> treat like Task 1's engines_agree
                            d.confidence = "high"
                            d.reason = f"{d.reason} (accepted: both models agree)"
                        decisions[k] = d
            used.add(second.name)
    return decisions, " + ".join(sorted(used))
