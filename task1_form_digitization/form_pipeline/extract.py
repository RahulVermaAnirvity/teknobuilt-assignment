"""Per-page orchestration: quality gate -> normalise -> locate -> read -> validate -> escalate."""
from __future__ import annotations

import difflib
import logging
import re
import time
from typing import Optional

import numpy as np

from . import config as C
from .checkbox import decide_multi, decide_single, ink_fraction
from .ingest import RawPage
from .layout import Anchors, Box, Grid, detect_grid, find_anchors
from .ocr import handwriting_engine, ink_bbox, page_words, printed_text
from .preprocess import binarize, blur_score, deskew, ink_ratio, rotate90
from .schema import (ChecklistItem, ChoiceResult, ExpectationItem, FormData,
                     PageResult, QualityReport, TextField)
from .validate import (has_strikethrough, validate_date, validate_free_text,
                       validate_name, validate_number)
from .vlm import MockVLM, VLMClient

log = logging.getLogger(__name__)

CWA_ROWS = [["1000", "2000", "3000", "4000", "8000"], ["5000", "5900", "6000", "7000", "OSLD"]]
CWA_CODES = {c for row in CWA_ROWS for c in row}
MAX_CHECKLIST_ROWS = 30


class PageRejected(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------- gate

def _quality_gate(raw: RawPage, q: QualityReport) -> None:
    gray = raw.image
    q.ink_ratio = round(ink_ratio(gray), 4)
    q.blur_score = round(blur_score(gray), 1)

    stamp = raw.text_layer.upper()
    hit = next((m for m in C.VOID_MARKERS if m in stamp), None)
    if hit:
        # A digital "VOID/FAILED TEST" stamp is an explicit instruction not to ingest.
        raise PageRejected(f"void_marker_in_text_layer:{hit.lower().replace(' ', '_')}")
    if q.ink_ratio < C.BLANK_INK_RATIO:
        raise PageRejected("blank_page")
    if q.ink_ratio > C.HEAVY_NOISE_INK_RATIO:
        q.warnings.append("heavy_noise_or_dark_scan")
    if q.blur_score < C.BLUR_LAPLACIAN_VAR:
        q.warnings.append("blurry_scan")


def _has_grid(grid: Grid) -> bool:
    return len(grid.h) >= C.MIN_GRID_H_LINES and len(grid.v) >= C.MIN_GRID_V_LINES


def _normalise_orientation(gray: np.ndarray, q: QualityReport) -> tuple[np.ndarray, Grid, Anchors]:
    """Deskew, prove there is a ruled form (cheap), then find the upright orientation
    via the printed labels (expensive OCR - only spent on pages that look like forms)."""
    g0, angle = deskew(gray)
    grid0 = detect_grid(binarize(g0))
    q.grid_h_lines, q.grid_v_lines = len(grid0.h), len(grid0.v)
    if not _has_grid(grid0):
        # A rule grid is rotation-symmetric (h/v swap), so no need to try 90/180/270 here.
        raise PageRejected("form_grid_not_found")

    best = None
    for turns in (0, 2, 1, 3):  # upright first, then upside-down (most common mis-feed)
        g = rotate90(g0, turns)
        if turns % 2:
            g, _ = deskew(g)
        anchors = find_anchors(page_words(g))
        if best is None or len(anchors.found) > len(best[1].found):
            best = (g, anchors, turns)
        if len(anchors.found) >= C.MIN_ANCHORS_FOUND + 3:
            break
    g, anchors, turns = best
    q.deskew_angle_deg = round(angle, 2)
    q.rotation_applied_deg = turns * 90
    q.anchors_found = sorted(anchors.found)
    grid = grid0 if turns == 0 else detect_grid(binarize(g))
    q.grid_h_lines, q.grid_v_lines = len(grid.h), len(grid.v)
    return g, grid, anchors


# --------------------------------------------------------------------------- helpers

def _clean_printed(text: str) -> str:
    text = text.replace("�", "-").replace("|", "")
    return re.sub(r"\s+", " ", text).strip(" -")


def _printed_field(gray: np.ndarray, box: Optional[Box]) -> TextField:
    if box is None:
        return TextField(flags=["cell_not_found"])
    text, conf = printed_text(box.inset(6).crop(gray))
    text = _clean_printed(text)
    ok = bool(text) and conf * 100 >= C.PRINTED_ACCEPT_CONF
    return TextField(value=text or None, raw=text or None, confidence=round(conf, 3), engine="tesseract",
                     needs_review=not ok, flags=[] if ok else (["low_confidence"] if text else ["empty"]))


def _canonical_title(field: TextField) -> TextField:
    """Snap the OCR'd checklist title onto the registry of known form types.

    Titles come from a closed set of printed form variants, so an OCR typo
    ("Savina" for "Saving") should resolve to the canonical name, and a title we
    have never seen is a new/unknown form layout that a human should look at.
    """
    if not field.raw:
        return field
    key = re.sub(r"[^a-z0-9]", "", field.raw.lower())
    best, score = None, 0.0
    for title in C.KNOWN_FORM_TITLES:
        s = difflib.SequenceMatcher(None, key, re.sub(r"[^a-z0-9]", "", title.lower())).ratio()
        if s > score:
            best, score = title, s
    if score >= C.FORM_TITLE_MATCH:
        field.value, field.needs_review = best, False
        field.flags = [f for f in field.flags if f != "low_confidence"]
    else:
        field.value, field.needs_review = None, True
        field.flags.append("unknown_form_type")
    return field


def _row_scores(gray: np.ndarray, grid: Grid, row: Box, cols: list[Box], codes) -> tuple[dict, bool]:
    """Ink score for each answer box on this row, plus whether the boxes really line up
    with the row. If a faint rule vanished on one side of the table (blur, fold, mud),
    the question cell spans two answer rows; reading either would silently attach an
    answer to the wrong question, so we report misalignment instead."""
    scores, aligned = {}, True
    for code, col in zip(codes, cols):
        box = grid.cell_at(col.cx, row.cy)
        if box is None:
            scores[code], aligned = 0.0, False
            continue
        overlap = min(box.y1, row.y1) - max(box.y0, row.y0)
        if overlap < C.ROW_ALIGN_MIN * max(box.h, row.h):
            aligned = False
        scores[code] = ink_fraction(gray, box)
    return scores, aligned


def _misaligned(res):
    res.value, res.needs_review = None, True
    res.flags.append("row_alignment_failed")
    return res


def _value_box(grid: Grid, anchor: Optional[Box]) -> Optional[Box]:
    """Cell immediately right of the label cell containing `anchor`.

    Vertical bounds come from the label cell (printed side, never written on), so a
    strike-through stroke inside the value cell can't masquerade as a table rule.
    """
    if anchor is None:
        return None
    label = grid.cell_at(anchor.cx, anchor.cy)
    if label is None:
        return None
    nxt = grid.right_neighbour(label)
    if nxt is None:
        return None
    return Box(label.x1, label.y0, nxt.x1, label.y1)


def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


class _Reader:
    def __init__(self, gray: np.ndarray, vlm: VLMClient, degraded: bool = False):
        self.gray = gray
        self.vlm = vlm
        # On a blurry / noisy page, model confidence is not calibrated: nothing handwritten is auto-accepted.
        self.degraded = degraded
        self.hw = handwriting_engine()
        self.crops: dict[str, np.ndarray] = {}

    def handwritten(self, name: str, box: Optional[Box], validator, multiline=False) -> TextField:
        if box is None:
            return TextField(flags=["cell_not_found"])
        cell = box.inset(8).crop(self.gray)
        bb = ink_bbox(cell)
        if bb is None:
            return validator("", 1.0, self.hw.engine)
        crop = bb.crop(cell)
        self.crops[name] = crop
        text, conf, alt = (self.hw.read_region(crop) if multiline else self.hw.read_line(crop))
        field = validator(text, conf, self.hw.engine)
        if alt:
            field.alternatives.append(alt)
            field.flags.append("ambiguous_reading")
            field.needs_review = True
        if self.degraded:
            field.flags.append("degraded_scan")
            field.needs_review = True
        if has_strikethrough(crop):
            field.flags.append("strikethrough_detected")
            field.needs_review = True
        return self.escalate(name, field, validator)

    def escalate(self, name: str, field: TextField, validator) -> TextField:
        """Second opinion from the VLM tier for fields we couldn't accept locally.

        Acceptance policy (a VLM is a second reader, not an oracle):
          * VLM agrees with the local reading (or its runner-up beam) -> two independent
            engines agree -> accept;
          * local reading was invalid, VLM is highly confident, page not degraded -> accept,
            tagged `vlm_only` so it can be audited;
          * otherwise (disagreement, low confidence, strike-through) -> human review,
            with both readings kept as alternatives.
        """
        if not field.needs_review or "empty" in field.flags or isinstance(self.vlm, MockVLM):
            return field
        res = self.vlm.transcribe(self.crops[name], name)
        if not res or not res.get("legible") or not res.get("text"):
            field.flags.append("vlm_could_not_read" if res else "vlm_unavailable")
            return field
        conf = {"high": 0.95, "medium": 0.85, "low": 0.6}[res["confidence"]]
        new = validator(res["text"], conf, self.vlm.name)
        new.flags.append(f"escalated_from:{field.engine}")
        local_readings = [r for r in [field.value, field.raw, *field.alternatives] if r]
        new.alternatives = [r for r in dict.fromkeys(local_readings) if _canon(r) != _canon(new.raw or "")]

        struck = res.get("struck_through_present") or "strikethrough_detected" in field.flags
        agree = bool(new.value) and any(_canon(new.value) == _canon(r) for r in local_readings)
        if struck:
            new.flags.append("strikethrough_detected")
            new.needs_review = True
        elif agree:
            new.flags.append("engines_agree")
            new.needs_review = bool([f for f in new.flags if f not in ("engines_agree",) and not f.startswith("escalated_from")])
        elif field.value is None and new.value and res["confidence"] == "high" and not self.degraded:
            new.flags.append("vlm_only")
            new.needs_review = False
        else:
            new.flags.append("engines_disagree")
            new.needs_review = True
        return new


# --------------------------------------------------------------------------- sections

def _read_location(gray: np.ndarray, grid: Grid, anchors: Anchors):
    a = anchors.get("location")
    label = grid.cell_at(a.cx, a.cy) if a else None
    if label is None:
        return decide_multi({})
    scores: dict[str, float] = {}
    for r, frac in enumerate((0.25, 0.75)):
        y = int(label.y0 + frac * label.h)
        cells, x = [], label.x1 + 2 * grid.tol
        while len(cells) < 12:
            c = grid.cell_at(x, y)
            if c is None:
                break
            cells.append(c)
            x = c.x1 + 2 * grid.tol
        for i in range(0, len(cells) - 1, 2):
            code_cell, box_cell = cells[i], cells[i + 1]
            txt, _ = printed_text(code_cell.inset(5).crop(gray), psm=7)
            code = re.sub(r"[^0-9A-Z]", "", txt.upper())
            if code not in CWA_CODES:  # OCR miss -> fall back to the template's printed order
                code = CWA_ROWS[r][i // 2] if r < 2 and i // 2 < len(CWA_ROWS[r]) else f"r{r}c{i // 2}"
            scores[code] = ink_fraction(gray, box_cell)
    return decide_multi(scores)


def _read_shift(gray: np.ndarray, grid: Grid, anchors: Anchors):
    scores = {}
    for name in ("day", "night"):
        a = anchors.get(name)
        cell = grid.cell_at(a.cx, a.cy) if a else None
        box = grid.right_neighbour(cell) if cell else None
        if box is not None:
            scores[name] = ink_fraction(gray, box)
    res = decide_single(scores, allow_none=False)
    if len(scores) < 2:
        res.flags.append("shift_box_missing")
        res.needs_review = True
    return res


def _read_checklist(gray: np.ndarray, grid: Grid, anchors: Anchors):
    a = anchors.get("safe")
    header = grid.cell_at(a.cx, a.cy) if a else None
    if header is None:
        return TextField(flags=["cell_not_found"]), [], ["checklist_header_not_found"]
    title_cell = grid.cell_at(gray.shape[1] // 2, header.y0 - 3 * grid.tol, max_size=gray.shape[1])
    title = _canonical_title(_printed_field(gray, title_cell))

    cols, c = [], header
    for _ in range(4):
        c = grid.right_neighbour(c)
        if c is None:
            break
        cols.append(c)
    if len(cols) != 4:
        return title, [], [f"expected_4_response_columns_found_{len(cols)}"]

    items, y = [], header.y1 + 2 * grid.tol
    heights: list[int] = []
    for i in range(MAX_CHECKLIST_ROWS):
        q = grid.cell_at(header.cx, y)
        if q is None or (heights and q.h > 3 * np.median(heights)):
            break  # end of the ruled checklist (blank band below)
        heights.append(q.h)
        scores, aligned = _row_scores(gray, grid, q, cols, "SRIN")
        question = _printed_field(gray, q)
        resp = decide_single(scores, allow_none=True)
        if not aligned:
            resp = _misaligned(resp)
        y = q.y1 + 2 * grid.tol
        if question.value is None and resp.value is None and "multiple_marks" not in resp.flags:
            continue  # unused template row
        if question.value and "no_mark" in resp.flags:
            resp.needs_review = True  # question printed but nobody answered it
        items.append(ChecklistItem(item_no=len(items) + 1, question=question, response=resp))
    return title, items, []


def _read_expectations(gray: np.ndarray, grid: Grid, anchors: Anchors):
    a = anchors.get("cczjv")
    header = grid.cell_at(a.cx, a.cy) if a else None
    if header is None:
        return [], ["expectation_header_not_found"]
    cols, c = [], header
    for _ in range(3):
        c = grid.right_neighbour(c)
        if c is None:
            break
        cols.append(c)
    if len(cols) != 3:
        return [], [f"expected_3_yes_no_na_columns_found_{len(cols)}"]
    stop_y = anchors.get("comments").y0 if anchors.get("comments") else gray.shape[0]
    found: dict[str, ChoiceResult] = {}
    warnings: list[str] = []
    row, steps = grid.below(header), 0
    while row is not None and row.cy < stop_y and steps < 10:
        steps += 1
        q = _printed_field(gray, row)
        # The four expectation questions are fixed printed text: identify each row by
        # its content, so a merged/skipped row can't shift answers onto the wrong question.
        canon = _match_text(q.raw or "", C.EXPECTATION_QUESTIONS)
        if canon is None or canon in found:
            warnings.append("unrecognised_expectation_row")
        else:
            scores, aligned = _row_scores(gray, grid, row, cols, ("yes", "no", "n/a"))
            res = decide_single(scores, allow_none=False)
            found[canon] = _misaligned(res) if not aligned else res
        row = grid.below(row)
    out = []
    for question in C.EXPECTATION_QUESTIONS:
        res = found.get(question) or ChoiceResult(flags=["question_row_not_found"])
        out.append(ExpectationItem(question=question, response=res))
    return out, sorted(set(warnings))


def _match_text(text: str, choices: list[str], min_ratio: float = 0.8) -> Optional[str]:
    key = re.sub(r"[^a-z0-9]", "", text.lower())
    best = max(choices, key=lambda c: difflib.SequenceMatcher(None, key, re.sub(r"[^a-z0-9]", "", c.lower())).ratio())
    ratio = difflib.SequenceMatcher(None, key, re.sub(r"[^a-z0-9]", "", best.lower())).ratio()
    return best if ratio >= min_ratio else None


def _comments_box(grid: Grid, anchors: Anchors) -> Optional[Box]:
    a = anchors.get("comments")
    if a is None:
        return None
    cell = grid.cell_at(a.cx, a.cy)
    if cell is None:
        return None
    return Box(cell.x0, a.y1 + 4, cell.x1, cell.y1)


# --------------------------------------------------------------------------- entry

def process_page(raw: RawPage, vlm: VLMClient) -> PageResult:
    t0 = time.perf_counter()
    q = QualityReport()
    result = PageResult(page_number=raw.page_number, page_sha1=raw.sha1, status="ok", quality=q)
    try:
        if raw.error or raw.image is None:
            result.status, result.rejection_reason = "error", raw.error or "no image"
            return result
        _quality_gate(raw, q)
        gray, grid, anchors = _normalise_orientation(raw.image, q)
        if len(anchors.found) < C.MIN_ANCHORS_FOUND:
            raise PageRejected("form_labels_not_found")

        rd = _Reader(gray, vlm, degraded=any(w in q.warnings for w in ("blurry_scan", "heavy_noise_or_dark_scan")))
        title, checklist, w1 = _read_checklist(gray, grid, anchors)
        expectations, w2 = _read_expectations(gray, grid, anchors)
        q.warnings += w1 + w2
        result.form = FormData(
            observation_date=rd.handwritten("observation_date", _value_box(grid, anchors.get("observation_date")), validate_date),
            observer=rd.handwritten("observer", _value_box(grid, anchors.get("observer")), validate_name),
            company_observed=rd.handwritten("company_observed", _value_box(grid, anchors.get("company_observed")), validate_name),
            location_cwa=_read_location(gray, grid, anchors),
            shift=_read_shift(gray, grid, anchors),
            timesheet_number=rd.handwritten("timesheet_number", _value_box(grid, anchors.get("timesheet")), validate_number),
            employee_number=rd.handwritten("employee_number", _value_box(grid, anchors.get("employee")), validate_number),
            checklist_title=title,
            checklist=checklist,
            expectation_questions=expectations,
            comments=rd.handwritten("comments", _comments_box(grid, anchors), validate_free_text, multiline=True),
        )
        result.review_fields = _review_fields(result.form)
        if result.review_fields or q.warnings:
            result.status = "needs_review"
    except PageRejected as rej:
        result.status, result.rejection_reason, result.form = "rejected", rej.reason, None
    except Exception as exc:  # last line of defence: one bad page never kills the batch
        log.exception("page %s failed", raw.page_number)
        result.status, result.rejection_reason, result.form = "error", f"{type(exc).__name__}: {exc}", None
    finally:
        result.processing_ms = int((time.perf_counter() - t0) * 1000)
    return result


def _review_fields(form: FormData) -> list[str]:
    out = []
    for name in ("observation_date", "observer", "company_observed", "location_cwa", "shift",
                 "timesheet_number", "employee_number", "checklist_title", "comments"):
        if getattr(form, name).needs_review:
            out.append(name)
    for it in form.checklist:
        if it.response.needs_review or it.question.needs_review:
            out.append(f"checklist[{it.item_no}]")
    for i, it in enumerate(form.expectation_questions, 1):
        if it.response.needs_review:
            out.append(f"expectation_questions[{i}]")
    return out
