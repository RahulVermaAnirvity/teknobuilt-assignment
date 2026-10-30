"""Field normalisation and validation - the anti-hallucination layer.

Rule: a value only reaches `TextField.value` if it is (a) read with enough
confidence and (b) passes a type-specific validator. Otherwise `value` is None,
the raw reading is preserved in `raw`, and the field is flagged for review.
We never "repair" a reading into something plausible-looking.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

import cv2
import numpy as np

from .config import FIELD_ACCEPT_CONF
from .schema import TextField

# Common handwriting confusions, applied ONLY to fields that must be numeric.
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "|": "1",
                            "Z": "2", "z": "2", "S": "5", "s": "5", "B": "8", "g": "9", "q": "9"})


def _clean(text: str) -> str:
    text = re.sub(r"\s+([.,;:!?])", r"\1", text or "")  # "followed ." -> "followed."
    return re.sub(r"\s+", " ", text).strip()


def _field(raw: str, conf: float, engine: str, value: str | None, flags: list[str]) -> TextField:
    ok = value is not None and conf >= FIELD_ACCEPT_CONF and not flags
    if value is not None and conf < FIELD_ACCEPT_CONF:
        flags = flags + ["low_confidence"]
    return TextField(value=value, raw=raw or None, confidence=round(conf, 3), engine=engine,
                     needs_review=not ok, flags=flags)


def validate_date(raw: str, conf: float, engine: str) -> TextField:
    s = _clean(raw).replace(" ", "")
    flags: list[str] = []
    value = None
    fmts = ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%m/%d/%Y", "%d/%m/%y", "%m/%d/%y"]
    parsed = {}
    for f in fmts:
        try:
            parsed[f] = datetime.strptime(s, f).date()
        except ValueError:
            pass
    if not s:
        flags.append("empty")
    elif not parsed:
        flags.append("unparseable_date")
    else:
        candidates = set(parsed.values())
        if len(candidates) > 1:
            flags.append("ambiguous_day_month")  # e.g. 03/04/2024
        d = next(iter(parsed.values()))
        if not (date(2000, 1, 1) <= d <= date.today() + timedelta(days=1)):
            flags.append("date_out_of_range")
        else:
            value = d.isoformat()
    return _field(raw, conf, engine, value if "ambiguous_day_month" not in flags else None, flags)


def validate_number(raw: str, conf: float, engine: str, min_len: int = 3, max_len: int = 10) -> TextField:
    s = _clean(raw).replace(" ", "").rstrip(".")
    flags: list[str] = []
    fixed = s.translate(_DIGIT_FIX)
    if fixed != s and fixed.isdigit():
        flags.append("char_substitution")  # we changed letters to digits; a human should confirm
    value = fixed if fixed.isdigit() and min_len <= len(fixed) <= max_len else None
    if not s:
        flags.append("empty")
    elif value is None:
        flags.append("not_a_number")
    return _field(raw, conf, engine, value, flags)


def validate_name(raw: str, conf: float, engine: str) -> TextField:
    s = _clean(raw).strip(" .,-")
    flags: list[str] = []
    value = s if re.fullmatch(r"[A-Za-z][A-Za-z .'&\-]{1,80}", s) else None
    if not s:
        flags.append("empty")
    elif value is None:
        flags.append("unexpected_characters")
    return _field(raw, conf, engine, value, flags)


def validate_free_text(raw: str, conf: float, engine: str, allow_empty: bool = True) -> TextField:
    s = _clean(raw)
    if not s:
        # an empty comments box is legitimate; it's not an extraction failure
        return TextField(value=None, raw=None, confidence=round(conf, 3), engine=engine,
                         needs_review=not allow_empty, flags=["empty"])
    return _field(raw, conf, engine, s, [])


def has_strikethrough(gray: np.ndarray) -> bool:
    """Detect a pen line crossing the written text (crossed-out entry).

    Looks for a long, thin horizontal ink run in the middle band of the ink
    bounding box. Table rules have already been cropped away by the caller.
    """
    if gray.size == 0:
        return False
    ink = (gray < 128).astype(np.uint8) * 255
    ys, xs = np.nonzero(ink)
    if len(xs) < 50:
        return False
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    band = ink[y0 + (y1 - y0) // 4: y1 - (y1 - y0) // 4 + 1, x0:x1 + 1]
    if band.size == 0:
        return False
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int((x1 - x0) * 0.6), 30), 1))
    return bool(cv2.morphologyEx(band, cv2.MORPH_OPEN, k).any())
