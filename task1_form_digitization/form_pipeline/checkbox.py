"""Deterministic checkbox reading.

Checkboxes don't need a language model: the question is "is there pen ink
inside this square?". We measure the dark-pixel fraction in the cell interior
(borders trimmed) and decide each single-select group *relative to its
siblings*, which cancels out page-wide effects like grey mud or toner.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import (CHECKBOX_FILLED, CHECKBOX_MARGIN, CHECKBOX_MARKED_MIN,
                     CHECKBOX_RELATIVE, INK_THRESHOLD)
from .layout import Box
from .schema import ChoiceResult


MIN_STROKE_PX = 12  # connected ink blobs smaller than this are dirt specks, not pen strokes


def ink_fraction(gray: np.ndarray, cell: Box) -> float:
    mx, my = int(cell.w * CHECKBOX_MARGIN), int(cell.h * CHECKBOX_MARGIN)
    roi = gray[cell.y0 + my:cell.y1 - my, cell.x0 + mx:cell.x1 - mx]
    if not roi.size:
        return 0.0
    mask = (roi < INK_THRESHOLD).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stroke_px = sum(int(a) for a in stats[1:, cv2.CC_STAT_AREA] if a >= MIN_STROKE_PX)
    return stroke_px / roi.size


def decide_single(scores: dict[str, float], allow_none: bool = True) -> ChoiceResult:
    """Pick the one marked option in a group, or flag why we can't."""
    res = ChoiceResult(scores={k: round(v, 4) for k, v in scores.items()})
    if not scores:
        res.flags.append("no_boxes_located")
        return res
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, s1), s2 = ranked[0], (ranked[1][1] if len(ranked) > 1 else 0.0)
    marked = [k for k, v in scores.items() if v >= CHECKBOX_MARKED_MIN]

    if s1 < CHECKBOX_MARKED_MIN:
        res.flags.append("no_mark")
        res.needs_review = not allow_none
        res.confidence = 1.0 - s1 / CHECKBOX_MARKED_MIN
        return res
    if any(scores[k] >= CHECKBOX_FILLED for k in marked):
        res.flags.append("box_filled_or_scribbled")
    if len(marked) > 1 and s1 < CHECKBOX_RELATIVE * max(s2, 1e-6):
        res.flags.append("multiple_marks")
        res.needs_review = True
        res.confidence = round(s1 / (s1 + s2), 3)
        return res

    if len(marked) > 1:
        res.flags.append("secondary_ink")  # stray mark / mud in a sibling box, winner still clear
    res.value = best
    res.confidence = round(min(1.0, (s1 - s2) / s1) if s1 else 0.0, 3)
    res.needs_review = bool(res.flags) or res.confidence < 0.6
    return res


def decide_multi(scores: dict[str, float]) -> ChoiceResult:
    """Location (CWA) is printed as a grid of boxes; normally one is ticked, but a
    worker covering two areas may tick two, so we return all and flag it."""
    res = decide_single(scores)
    if "multiple_marks" in res.flags:
        marked = sorted(k for k, v in scores.items() if v >= CHECKBOX_MARKED_MIN)
        res.value = ",".join(marked)
    return res
