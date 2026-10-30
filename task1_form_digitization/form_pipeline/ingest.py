"""Turn an input file into a stream of page images, one failure at a time.

A corrupted page must never take the whole batch down, so every page is
rendered inside its own try/except and errors are yielded as data.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import pymupdf

from .config import RENDER_DPI

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class RawPage:
    page_number: int                 # 1-based
    image: Optional[np.ndarray]      # grayscale uint8, None if rendering failed
    text_layer: str = ""             # embedded PDF text (digital stamps like "FAILED TEST FORM")
    embedded_images: int = 0
    error: Optional[str] = None

    @property
    def sha1(self) -> Optional[str]:
        return hashlib.sha1(self.image.tobytes()).hexdigest() if self.image is not None else None


def iter_pages(path: str | Path) -> Iterator[RawPage]:
    path = Path(path)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        yield RawPage(1, img, error=None if img is not None else "unreadable image file")
        return

    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # truncated / not a PDF / wrong password
        yield RawPage(1, None, error=f"cannot open document: {exc}")
        return
    if doc.needs_pass:
        yield RawPage(1, None, error="document is encrypted")
        return

    for idx in range(doc.page_count):
        yield _render(doc, idx)


def _render(doc, idx: int) -> RawPage:
    try:
        page = doc[idx]
        pix = page.get_pixmap(dpi=RENDER_DPI, colorspace=pymupdf.csGRAY, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w).copy()
        return RawPage(idx + 1, img, page.get_text("text") or "", len(page.get_images()))
    except Exception as exc:
        return RawPage(idx + 1, None, error=f"render failed: {exc}")


def count_pages(path: str | Path) -> int:
    """Number of pages iter_pages() will yield (1 for images and unreadable files)."""
    path = Path(path)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return 1
    try:
        doc = pymupdf.open(path)
        return 1 if doc.needs_pass else max(doc.page_count, 1)
    except Exception:
        return 1


def load_page(path: str | Path, page_number: int) -> RawPage:
    """Render exactly one page - what a Celery worker does for its task.

    Uses the same code path as iter_pages(), so a page processed by a worker is
    pixel-identical to the same page processed in a single local run.
    """
    path = Path(path)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return next(iter_pages(path))
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        return RawPage(page_number, None, error=f"cannot open document: {exc}")
    if doc.needs_pass:
        return RawPage(page_number, None, error="document is encrypted")
    if not 1 <= page_number <= doc.page_count:
        return RawPage(page_number, None, error=f"page {page_number} out of range (1..{doc.page_count})")
    return _render(doc, page_number - 1)
