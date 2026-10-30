"""Digitise handwritten safety observation forms into structured JSON.

Usage:
    python run_pipeline.py ../Sample_input.pdf -o ../outputs/task1_output.json
    python run_pipeline.py ../Sample_input.pdf -o out.json --vlm      # escalate unclear fields to Gemini/Claude
    python run_pipeline.py ../Sample_input.pdf -o out.json --celery   # one page per Celery worker
                                                                      # (start workers: python start_workers.py -n 4)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("USE_TF", "0")  # transformers: torch only, don't drag in TensorFlow
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from form_pipeline import __version__
from form_pipeline.extract import process_page
from form_pipeline.config import HANDWRITING_MODEL, PAGE_TIME_LIMIT_S
from form_pipeline.ingest import count_pages, iter_pages, load_page
from form_pipeline.ocr import handwriting_engine, tesseract_available
from form_pipeline.schema import PageResult, RunOutput, RunSummary
from form_pipeline.vlm import make_vlm


def _log(res: PageResult) -> None:
    logging.info("page %d -> %s %s (%d ms)", res.page_number, res.status,
                 res.rejection_reason or ",".join(res.review_fields), res.processing_ms)


def _finalise(path: Path, pages: list[PageResult], vlm_name: str, handwriting: str, mode: str,
              wall_s: float) -> RunOutput:
    """Shared by local and Celery runs: page order, duplicate detection, summary."""
    pages = sorted(pages, key=lambda p: p.page_number)
    seen: dict[str, int] = {}
    for res in pages:
        # The same physical sheet scanned twice should not create two observations.
        if res.page_sha1 and res.page_sha1 in seen and res.status != "rejected":
            res.status, res.rejection_reason = "rejected", f"duplicate_of_page_{seen[res.page_sha1]}"
            res.form = None
        elif res.page_sha1:
            seen[res.page_sha1] = res.page_number
    count = lambda s: sum(p.status == s for p in pages)  # noqa: E731
    return RunOutput(
        source_file=path.name,
        pipeline_version=__version__,
        engines={"layout": "opencv-grid+tesseract-anchors", "printed_text": "tesseract",
                 "handwriting": handwriting, "checkboxes": "ink-density",
                 "vlm_escalation": vlm_name, "execution": f"{mode} ({wall_s:.1f}s wall-clock)"},
        processed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        summary=RunSummary(pages_total=len(pages), ok=count("ok"), needs_review=count("needs_review"),
                           rejected=count("rejected"), error=count("error")),
        pages=pages,
    )


def run(path: Path, use_vlm: bool, on_page=None) -> RunOutput:
    """Process every page in this process, one after another.
    `on_page(raw_page, result)` is called after each page (Streamlit progress/previews)."""
    t0 = time.perf_counter()
    vlm = make_vlm(use_vlm)
    pages = []
    for raw in iter_pages(path):
        res = process_page(raw, vlm)
        _log(res)
        pages.append(res)
        if on_page:
            on_page(raw, res)
    return _finalise(path, pages, vlm.name, handwriting_engine().engine, "local, 1 page at a time",
                     time.perf_counter() - t0)


def celery_workers_online(timeout: float = 2.5) -> int:
    """How many pages Celery can process at the same time: the sum of every online
    worker's pool size (0 if no worker answers / broker unreachable)."""
    try:
        from form_pipeline.tasks import app
        stats = app.control.inspect(timeout=timeout).stats() or {}
    except Exception:
        return 0
    return sum(int((s.get("pool") or {}).get("max-concurrency", 1)) for s in stats.values())


def run_celery(path: Path, use_vlm: bool, on_page=None, poll_s: float = 0.5) -> RunOutput:
    """Fan the document out as one Celery task per page and gather the results.

    Workers run the unchanged process_page() on a pixel-identical render of their
    page, so output equals a local run - only faster. A task that fails or exceeds
    the time limit becomes an `error` page instead of failing the batch.
    """
    from form_pipeline.tasks import process_page_task

    t0 = time.perf_counter()
    path = Path(path).resolve()  # workers open the file themselves
    n = count_pages(path)
    pending = {i: process_page_task.delay(str(path), i, use_vlm) for i in range(1, n + 1)}
    logging.info("dispatched %d page task(s) to Celery", n)
    deadline = time.monotonic() + PAGE_TIME_LIMIT_S + 60 * n
    pages: list[PageResult] = []
    while pending:
        for i, ar in list(pending.items()):
            if not ar.ready():
                if time.monotonic() > deadline:
                    ar.revoke()
                    res = PageResult(page_number=i, status="error", rejection_reason="celery: no worker finished the page in time")
                else:
                    continue
            elif ar.successful():
                res = PageResult.model_validate(ar.get())
            else:
                res = PageResult(page_number=i, status="error", rejection_reason=f"celery task failed: {ar.result!r}"[:300])
            del pending[i]
            _log(res)
            pages.append(res)
            if on_page:
                on_page(load_page(path, i), res)
        if pending:
            time.sleep(poll_s)
    vlm_name = make_vlm(use_vlm).name  # no API call; just the configured provider
    return _finalise(path, pages, vlm_name, f"trocr:{HANDWRITING_MODEL}", "celery, 1 page per task",
                     time.perf_counter() - t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="PDF or image file")
    ap.add_argument("-o", "--output", type=Path, default=Path("output.json"))
    ap.add_argument("--vlm", action="store_true", help="escalate low-confidence fields to a vision model (Gemini via GEMINI_API_KEY, or Claude via ANTHROPIC_API_KEY)")
    ap.add_argument("--celery", action="store_true", help="process pages in parallel on Celery workers, one page per task")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if not args.input.exists():
        logging.error("input not found: %s", args.input)
        return 2
    if not tesseract_available():
        logging.error("tesseract binary not found; install it or set TESSERACT_CMD")
        return 2

    if args.celery:
        workers = celery_workers_online()
        if not workers:
            logging.error("no Celery workers are running - start them first:  python start_workers.py -n 4")
            return 3
        logging.info("Celery online: %d page(s) can be processed at a time", workers)
        out = run_celery(args.input, args.vlm)
    else:
        out = run(args.input, args.vlm)
    logging.info("execution: %s", out.engines["execution"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("summary: %s -> %s", out.summary.model_dump(), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
