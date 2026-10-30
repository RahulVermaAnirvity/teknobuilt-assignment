"""Robustness harness: degrade the sample pages synthetically and check the pipeline
either reproduces the clean extraction or refuses / flags - never silently returns
different data.

    python stress_test.py ../Sample_input.pdf -o ../outputs/task1_stress_report.json

A case PASSES when every field is either identical to the clean run, or explicitly
flagged (needs_review / rejected). A case FAILS when a value differs from the
clean run while being marked as trustworthy - i.e. a silent hallucination.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
warnings.filterwarnings("ignore")

import cv2
import numpy as np

from form_pipeline.extract import process_page
from form_pipeline.ingest import RawPage, iter_pages
from form_pipeline.preprocess import rotate
from form_pipeline.vlm import make_vlm

RNG = np.random.default_rng(7)


# ------------------------------------------------------------------ degradations

def skew(deg):
    return lambda g: rotate(g, deg)


def upside_down(g):
    return np.ascontiguousarray(g[::-1, ::-1])


def mud(g):
    """Translucent brown-grey stains + speckle, like a form dropped on site."""
    out = g.astype(np.float32)
    h, w = g.shape
    for _ in range(9):
        mask = np.zeros_like(g, np.float32)
        c = (int(RNG.integers(0, w)), int(RNG.integers(0, h)))
        axes = (int(RNG.integers(60, 260)), int(RNG.integers(40, 160)))
        cv2.ellipse(mask, c, axes, float(RNG.integers(0, 180)), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (61, 61), 0)
        out = out * (1 - 0.45 * mask) + 150 * 0.45 * mask  # muddy grey ~150, lighter than pen ink
    speck = RNG.random(g.shape) < 0.002
    out[speck] = 60
    return np.clip(out, 0, 255).astype(np.uint8)


def blur(g):
    return cv2.GaussianBlur(g, (5, 5), 0)


def low_res_fax(g):
    h, w = g.shape
    small = cv2.resize(g, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def strike_observer(g):
    """Cross out the observer's name with a pen line (value cell ~ y 110, x 600-1000 @200dpi)."""
    out = g.copy()
    cv2.line(out, (640, 108), (960, 116), 20, 3)
    return out


def blank(g):
    return np.full_like(g, 250)


def black(g):
    return np.full_like(g, 15)


def noise(g):
    return (RNG.random(g.shape) * 255).astype(np.uint8)


CASES = [
    ("skew_+4deg", 0, skew(4)), ("skew_-7deg", 1, skew(-7)), ("upside_down", 2, upside_down),
    ("mud_stains", 0, mud), ("mud_plus_skew", 1, lambda g: mud(skew(3)(g))), ("blur", 2, blur),
    ("low_res_fax", 0, low_res_fax), ("crossed_out_observer", 1, strike_observer),
    ("blank_page", 0, blank), ("black_page", 0, black), ("random_noise", 0, noise),
]


# ------------------------------------------------------------------ comparison

def flatten(form: dict | None) -> dict[str, tuple]:
    """field -> (value, needs_review) for everything downstream would consume."""
    if not form:
        return {}
    out = {}
    for k in ("observation_date", "observer", "company_observed", "timesheet_number",
              "employee_number", "checklist_title", "comments", "location_cwa", "shift"):
        out[k] = (form[k]["value"], form[k]["needs_review"])
    for it in form["checklist"]:
        out[f"checklist[{it['item_no']}]"] = (it["response"]["value"], it["response"]["needs_review"])
    for it in form["expectation_questions"]:
        out[f"expectation[{it['question'][:30]}]"] = (it["response"]["value"], it["response"]["needs_review"])
    return out


def judge(clean: dict, got: dict, expect_reject: bool) -> dict:
    if expect_reject:
        ok = got["status"] in ("rejected", "error")
        return {"pass": ok, "detail": got.get("rejection_reason")}
    if got["status"] in ("rejected", "error"):
        return {"pass": True, "detail": f"refused: {got.get('rejection_reason')}", "matched": 0, "flagged": 0}
    c, g = flatten(clean["form"]), flatten(got["form"])
    silent_wrong, flagged, matched = [], [], 0
    for k, (cv, _) in c.items():
        gv, review = g.get(k, (None, True))
        if gv == cv:
            matched += 1
        elif review or gv is None:
            flagged.append(k)
        else:
            silent_wrong.append(f"{k}: {cv!r} -> {gv!r}")
    return {"pass": not silent_wrong, "matched": matched, "flagged": flagged,
            "silent_wrong": silent_wrong, "total": len(c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("stress_report.json"))
    ap.add_argument("--vlm", action="store_true", help="use the real VLM escalation tier (Gemini/Claude)")
    args = ap.parse_args()
    vlm = make_vlm(args.vlm)
    print(f"VLM tier: {getattr(vlm, 'name', 'mock')}")

    pages = [p for p in iter_pages(args.pdf)]
    forms = [p for p in pages if p.embedded_images]
    clean = {i: process_page(p, vlm).model_dump(mode="json") for i, p in enumerate(forms)}
    report = []

    for name, idx, fn in CASES:
        src = forms[idx]
        raw = RawPage(src.page_number, fn(src.image))
        got = process_page(raw, vlm).model_dump(mode="json")
        verdict = judge(clean[idx], got, expect_reject=name in ("blank_page", "black_page", "random_noise"))
        report.append({"case": name, "source_page": src.page_number, "status": got["status"],
                       "processing_ms": got["processing_ms"],
                       "quality": {k: got["quality"][k] for k in ("deskew_angle_deg", "rotation_applied_deg", "warnings")},
                       **verdict})
        print(f"{'PASS' if verdict['pass'] else 'FAIL'}  {name:22s} status={got['status']:12s} "
              f"{verdict.get('detail') or ''} matched={verdict.get('matched')}/{verdict.get('total')} "
              f"flagged={verdict.get('flagged')} {verdict.get('silent_wrong') or ''}")

    # The "FAILED TEST FORM" page as a pure scan (no digital text layer to give it away).
    failed = next(p for p in pages if not p.embedded_images)
    got = process_page(RawPage(failed.page_number, failed.image, text_layer=""), vlm).model_dump(mode="json")
    v = judge({}, got, expect_reject=True)
    report.append({"case": "failed_form_scanned_no_text_layer", "status": got["status"], **v})
    print(f"{'PASS' if v['pass'] else 'FAIL'}  failed_form_scanned     status={got['status']} {v['detail']}")

    # File-level corruption: truncated PDF and garbage bytes must not raise.
    data = args.pdf.read_bytes()
    with tempfile.TemporaryDirectory() as td:
        for case, payload in (("truncated_pdf", data[: len(data) // 3]), ("garbage_bytes", os.urandom(4096))):
            fp = Path(td) / f"{case}.pdf"
            fp.write_bytes(payload)
            try:
                res = [process_page(p, vlm) for p in iter_pages(fp)]
                statuses = [r.status for r in res]
                ok = all(s in ("rejected", "error") for s in statuses) or bool(res)
                detail = [(r.page_number, r.status, r.rejection_reason) for r in res][:4]
            except Exception as exc:  # this is what we're guarding against
                ok, statuses, detail = False, [], f"CRASH {exc!r}"
            report.append({"case": case, "pass": ok, "statuses": statuses, "detail": str(detail)})
            print(f"{'PASS' if ok else 'FAIL'}  {case:22s} {detail}")

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    passed = sum(r["pass"] for r in report)
    print(f"\n{passed}/{len(report)} cases passed -> {args.output}")


if __name__ == "__main__":
    main()
