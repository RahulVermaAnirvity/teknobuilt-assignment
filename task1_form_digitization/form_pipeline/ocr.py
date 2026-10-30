"""Text recognition engines.

* Tesseract  - printed text (labels, checklist questions, title). Fast, local, ~95% conf here.
* TrOCR      - handwritten single-line fields (names, dates, numbers, comments).
               Open-source transformer, runs on CPU, no per-call cost.

Both return (text, confidence in [0, 1]). If TrOCR can't be loaded (no torch,
offline without cached weights) we degrade to Tesseract and the low confidence
routes fields to review instead of crashing.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np
import pytesseract

from .config import HANDWRITING_MODEL, tesseract_cmd
from .layout import Box, Word

log = logging.getLogger(__name__)

_cmd = tesseract_cmd()
if _cmd:
    pytesseract.pytesseract.tesseract_cmd = _cmd


def tesseract_available() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def page_words(gray: np.ndarray, psm: int = 11) -> list[Word]:
    d = pytesseract.image_to_data(gray, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    out = []
    for i, txt in enumerate(d["text"]):
        if txt.strip() and float(d["conf"][i]) >= 0:
            b = Box(d["left"][i], d["top"][i], d["left"][i] + d["width"][i], d["top"][i] + d["height"][i])
            out.append(Word(txt.strip(), float(d["conf"][i]), b))
    return out


def printed_text(gray: np.ndarray, psm: int = 6) -> tuple[str, float]:
    if gray.size == 0:
        return "", 0.0
    d = pytesseract.image_to_data(gray, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    words = [(t.strip(), float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]
    if not words:
        return "", 0.0
    text = " ".join(w for w, _ in words)
    return text, float(np.mean([c for _, c in words])) / 100.0


def ink_bbox(gray: np.ndarray, thresh: int = 128, pad: int = 10) -> Optional[Box]:
    """Tight box around dark pixels, ignoring specks."""
    mask = (gray < thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ys, xs = np.nonzero(mask)
    if len(xs) < 15:
        return None
    h, w = gray.shape
    return Box(max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, w), min(ys.max() + pad, h))


def split_lines(gray: np.ndarray, thresh: int = 128, min_gap: int = 6) -> list[np.ndarray]:
    """Split a multi-line handwriting region into single text lines (TrOCR is line-level)."""
    rows = (gray < thresh).sum(axis=1) > 2
    lines, start, gap = [], None, 0
    for y, on in enumerate(rows):
        if on and start is None:
            start, gap = y, 0
        elif not on and start is not None:
            gap += 1
            if gap >= min_gap:
                lines.append((start, y - gap))
                start = None
    if start is not None:
        lines.append((start, len(rows)))
    return [gray[max(a - 6, 0):b + 6] for a, b in lines if b - a >= 10]


class HandwritingOCR:
    def __init__(self, model_name: str = HANDWRITING_MODEL):
        self.model_name = model_name
        self._loaded = False
        self._proc = self._model = None
        self.error: Optional[str] = None

    @property
    def engine(self) -> str:
        return f"trocr:{self.model_name}" if self.available else "tesseract-fallback"

    @property
    def available(self) -> bool:
        self._load()
        return self._model is not None

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            import torch  # noqa: F401
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
            self._proc = TrOCRProcessor.from_pretrained(self.model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name).eval()
        except Exception as exc:
            self.error = str(exc)
            log.warning("TrOCR unavailable (%s); falling back to tesseract for handwriting", exc)

    def read_line(self, gray: np.ndarray) -> tuple[str, float, Optional[str]]:
        """Returns (best text, confidence, competing reading or None).

        The sequence probability alone is over-confident on degraded input (a blurred
        '09' read as '08' still scores ~0.85), so we also keep the runner-up beam: if
        a materially different reading is nearly as likely, the field is ambiguous.
        """
        if gray.size == 0:
            return "", 0.0, None
        if not self.available:
            text, conf = printed_text(gray, psm=7)
            return text, conf, None
        import torch
        from PIL import Image
        rgb = Image.fromarray(gray).convert("RGB")
        pv = self._proc(images=rgb, return_tensors="pt").pixel_values
        with torch.no_grad():
            out = self._model.generate(pv, max_new_tokens=48, num_beams=5, num_return_sequences=2,
                                       output_scores=True, return_dict_in_generate=True)
        texts = [t.strip() for t in self._proc.batch_decode(out.sequences, skip_special_tokens=True)]
        # sequences_scores is the length-normalised log-prob of each returned beam
        p = torch.exp(out.sequences_scores).tolist()
        alt = None
        if len(texts) > 1 and _canon(texts[1]) != _canon(texts[0]) and p[1] / p[0] >= AMBIGUITY_RATIO:
            alt = texts[1]
        return texts[0], float(p[0]), alt

    def read_region(self, gray: np.ndarray) -> tuple[str, float, Optional[str]]:
        """Read a possibly multi-line region; confidence = weakest line."""
        lines = split_lines(gray)
        if not lines:
            return "", 0.0, None
        results = [self.read_line(l) for l in lines]
        alts = [a for _, _, a in results if a]
        return (" ".join(t for t, _, _ in results if t), min(c for _, c, _ in results),
                alts[0] if alts else None)


AMBIGUITY_RATIO = 0.80  # runner-up beam at >= 80% of the winner's probability -> ambiguous


def _canon(s: str) -> str:
    """Ignore differences that don't change the value (spacing, trailing punctuation, case)."""
    return re.sub(r"[\s.,;:]+", "", s).lower()


@lru_cache(maxsize=1)
def handwriting_engine() -> HandwritingOCR:
    return HandwritingOCR()
