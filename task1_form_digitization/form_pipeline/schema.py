"""Output contract for one digitised safety-observation page.

Every extracted value carries provenance (engine + confidence) and a
`needs_review` flag. A value that fails validation is never silently
"fixed": `value` is set to None and the raw reading is kept in `raw` so a
reviewer can see what the machine saw. Downstream systems should consume
`value` only.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

PageStatus = Literal["ok", "needs_review", "rejected", "error"]
CheckState = Literal["S", "R", "I", "N"]
YesNoNA = Literal["yes", "no", "n/a"]


class TextField(BaseModel):
    value: Optional[str] = None
    raw: Optional[str] = None
    confidence: float = 0.0
    engine: Optional[str] = None
    needs_review: bool = True
    flags: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)  # competing readings shown to the reviewer


class ChoiceResult(BaseModel):
    """Result of a single-select group of checkboxes (e.g. S/R/I/N)."""
    value: Optional[str] = None
    confidence: float = 0.0
    scores: dict[str, float] = Field(default_factory=dict)  # ink fraction per box
    needs_review: bool = True
    flags: list[str] = Field(default_factory=list)


class ChecklistItem(BaseModel):
    item_no: int
    question: TextField
    response: ChoiceResult  # value in CheckState


class ExpectationItem(BaseModel):
    question: str
    response: ChoiceResult  # value in YesNoNA


class FormData(BaseModel):
    observation_date: TextField
    observer: TextField
    company_observed: TextField
    location_cwa: ChoiceResult  # value = CWA code; multi-select is flagged
    shift: ChoiceResult
    timesheet_number: TextField
    employee_number: TextField
    checklist_title: TextField
    checklist: list[ChecklistItem]
    expectation_questions: list[ExpectationItem]
    comments: TextField


class QualityReport(BaseModel):
    ink_ratio: float = 0.0
    blur_score: float = 0.0
    deskew_angle_deg: float = 0.0
    rotation_applied_deg: int = 0
    grid_h_lines: int = 0
    grid_v_lines: int = 0
    anchors_found: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PageResult(BaseModel):
    page_number: int  # 1-based, as a human would cite it
    page_sha1: Optional[str] = None
    status: PageStatus
    rejection_reason: Optional[str] = None
    quality: QualityReport = Field(default_factory=QualityReport)
    form: Optional[FormData] = None
    review_fields: list[str] = Field(default_factory=list)
    processing_ms: int = 0


class RunSummary(BaseModel):
    pages_total: int
    ok: int
    needs_review: int
    rejected: int
    error: int


class RunOutput(BaseModel):
    source_file: str
    pipeline_version: str
    engines: dict[str, str]
    processed_at: str
    summary: RunSummary
    pages: list[PageResult]
