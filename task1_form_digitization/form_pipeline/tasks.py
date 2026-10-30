"""Celery workers: one page = one task.

Speed comes only from running pages side by side. Every task calls the exact
same `process_page()` as a local run, on a pixel-identical render of the page,
so the extracted data is the same as a single-process run.

Broker / result store
  * default (zero setup, single machine): kombu's filesystem broker in ./.celery
    and a SQLite result backend - works on Windows without Redis or Docker;
  * production: set CELERY_BROKER_URL=redis://host:6379/0 (and optionally
    CELERY_RESULT_BACKEND); workers can then run on any number of machines that
    can read the input files (shared volume / object storage).

Start workers with `python start_workers.py -n 4` (one solo-pool process per
worker, which is also the supported way to run Celery on Windows).
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from celery import Celery  # noqa: E402
from celery.signals import worker_init, worker_process_init  # noqa: E402

from . import config as C  # noqa: E402 - also loads .env

log = logging.getLogger(__name__)

BROKER_URL = os.getenv("CELERY_BROKER_URL", "").strip()
DATA_DIR = Path(os.getenv("CELERY_DATA_DIR", Path(__file__).resolve().parents[2] / ".celery"))

if BROKER_URL:
    backend = os.getenv("CELERY_RESULT_BACKEND") or (BROKER_URL if BROKER_URL.startswith("redis") else None)
    app = Celery("form_pipeline", broker=BROKER_URL, backend=backend)
else:
    for sub in ("queue", "processed", "control"):
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)
    app = Celery("form_pipeline", broker="filesystem://",
                 backend=os.getenv("CELERY_RESULT_BACKEND") or f"db+sqlite:///{(DATA_DIR / 'results.sqlite').as_posix()}")
    app.conf.broker_transport_options = {
        # producer and consumers share one folder = one queue
        "data_folder_in": str(DATA_DIR / "queue"),
        "data_folder_out": str(DATA_DIR / "queue"),
        "processed_folder": str(DATA_DIR / "processed"),
        "store_processed": False,
        # broadcast channel (ping / inspect). Must be absolute: kombu's default is a
        # relative "control" dir, so processes started from different folders can't see each other.
        "control_folder": str(DATA_DIR / "control"),
    }

app.conf.update(
    task_serializer="json", result_serializer="json", accept_content=["json"],
    worker_prefetch_multiplier=1,       # a worker reserves one page at a time -> even load
    task_acks_late=True,                # page is re-queued if a worker dies mid-page
    task_reject_on_worker_lost=True,
    task_time_limit=C.PAGE_TIME_LIMIT_S,
    result_expires=24 * 3600,
    worker_hijack_root_logger=False,
)


@lru_cache(maxsize=2)
def _vlm(use_vlm: bool):
    from .vlm import make_vlm
    return make_vlm(use_vlm)


def _warm(**_):
    """Load the handwriting model once per worker process, before the first page arrives."""
    threads = os.getenv("PIPELINE_TORCH_THREADS")
    if threads:
        import torch
        torch.set_num_threads(int(threads))  # share the CPU fairly between workers; affects speed only
    from .ocr import handwriting_engine
    handwriting_engine().available  # noqa: B018 - triggers the lazy load


worker_init.connect(_warm)          # solo pool: the worker process itself runs tasks
worker_process_init.connect(_warm)  # prefork pool (Linux): each child process


@app.task(name="forms.process_page")
def process_page_task(path: str, page_number: int, use_vlm: bool) -> dict:
    from .extract import process_page
    from .ingest import load_page
    raw = load_page(path, page_number)
    return process_page(raw, _vlm(use_vlm)).model_dump(mode="json")
