"""Structure-relative form layout.

The filled forms do NOT match the blank template pixel-for-pixel: the
checklist title, number of questions (8, 9, ... up to 15) and row heights vary
per form type, and a scan can be shifted, scaled or skewed. So instead of
hard-coded crop boxes we:

1. detect the ruled table grid (horizontal + vertical line segments),
2. find printed anchor labels with OCR ("Observer", "Shift", "CCZJV", ...),
3. resolve every field as "the cell to the right of / below an anchor".

This survives different checklist lengths and moderate distortion, and it fails
loudly (missing anchor) instead of silently cropping the wrong region.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def cx(self) -> int:
        return (self.x0 + self.x1) // 2

    @property
    def cy(self) -> int:
        return (self.y0 + self.y1) // 2

    def inset(self, px: int) -> "Box":
        return Box(self.x0 + px, self.y0 + px, self.x1 - px, self.y1 - px)

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[max(self.y0, 0):max(self.y1, 0), max(self.x0, 0):max(self.x1, 0)]


@dataclass
class Word:
    text: str
    conf: float
    box: Box


@dataclass
class Grid:
    h: np.ndarray  # (N, 3) rows of [x0, x1, y]
    v: np.ndarray  # (M, 3) rows of [x, y0, y1]
    tol: int = 6

    def cell_at(self, x: int, y: int, max_size: int = 10**6) -> Optional[Box]:
        """Smallest ruled rectangle enclosing (x, y), or None if not enclosed."""
        t = self.tol
        hs = self.h[(self.h[:, 0] - t <= x) & (self.h[:, 1] + t >= x)]
        vs = self.v[(self.v[:, 1] - t <= y) & (self.v[:, 2] + t >= y)]
        above, below = hs[hs[:, 2] < y - 2, 2], hs[hs[:, 2] > y + 2, 2]
        left, right = vs[vs[:, 0] < x - 2, 0], vs[vs[:, 0] > x + 2, 0]
        if not (len(above) and len(below) and len(left) and len(right)):
            return None
        box = Box(int(left.max()), int(above.max()), int(right.min()), int(below.min()))
        return box if box.w < max_size and box.h < max_size else None

    def right_neighbour(self, cell: Box) -> Optional[Box]:
        return self.cell_at(cell.x1 + 2 * self.tol, cell.cy)

    def left_neighbour(self, cell: Box) -> Optional[Box]:
        return self.cell_at(cell.x0 - 2 * self.tol, cell.cy)

    def below(self, cell: Box, x: Optional[int] = None) -> Optional[Box]:
        return self.cell_at(cell.cx if x is None else x, cell.y1 + 2 * self.tol)


def _segments(mask: np.ndarray, horizontal: bool) -> np.ndarray:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for x, y, w, h, _ in stats[1:]:
        if horizontal and h <= 15:
            out.append((x, x + w, y + h // 2))
        elif not horizontal and w <= 15:
            out.append((x + w // 2, y, y + h))
    return np.asarray(out, dtype=int).reshape(-1, 3)


def detect_grid(bw: np.ndarray) -> Grid:
    """Extract ruled lines with long, thin morphological openings.

    Handwriting, X marks and mud blobs are short or thick so they don't survive
    the opening; a small closing afterwards bridges breaks caused by folds or
    pen strokes crossing a rule.
    """
    h, w = bw.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 30, 20), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 60, 20)))
    hmask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    hmask = cv2.morphologyEx(hmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)))
    vmask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    vmask = cv2.morphologyEx(vmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25)))
    return Grid(_segments(hmask, True), _segments(vmask, False))


def grid_mask(bw: np.ndarray, grid: Grid) -> np.ndarray:
    """Binary image containing only the ruled lines (used to erase them before OCR)."""
    m = np.zeros_like(bw)
    for x0, x1, y in grid.h:
        cv2.line(m, (int(x0), int(y)), (int(x1), int(y)), 255, 7)
    for x, y0, y1 in grid.v:
        cv2.line(m, (int(x), int(y0)), (int(x), int(y1)), 255, 7)
    return m


# --- Anchors -----------------------------------------------------------------

ANCHORS = {
    "observation_date": ["observation"],
    "observer": ["observer"],
    "company_observed": ["company"],
    "location": ["location", "(cwa)"],
    "shift": ["shift"],
    "day": ["day"],
    "night": ["night"],
    "timesheet": ["timesheet"],
    "employee": ["employee"],
    "safe": ["safe"],
    "cczjv": ["cczjv", "expectation", "questions"],
    "comments": ["comments:"],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9:()]", "", s.lower())


@dataclass
class Anchors:
    found: dict[str, Box] = field(default_factory=dict)

    def get(self, name: str) -> Optional[Box]:
        return self.found.get(name)


def find_anchors(words: list[Word]) -> Anchors:
    """Fuzzy-match the printed labels. Topmost match wins (labels appear once)."""
    found: dict[str, Box] = {}
    for name, keys in ANCHORS.items():
        cands = []
        for wd in words:
            t = _norm(wd.text)
            if not t:
                continue
            score = max(difflib.SequenceMatcher(None, t, k).ratio() for k in keys)
            if score >= 0.8 or (name == "comments" and t.startswith("comment")):
                cands.append((wd.box.y0, wd.box))
        if cands:
            found[name] = min(cands, key=lambda c: c[0])[1]
    # "Observation" also fuzzily matches "Observer"; keep the one on its own row.
    od, ob = found.get("observation_date"), found.get("observer")
    if od and ob and abs(od.cy - ob.cy) < 10:
        found.pop("observer")
    return Anchors(found)
