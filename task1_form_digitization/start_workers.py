"""Start N Celery worker slots, each processing one page at a time.

    python start_workers.py -n 4          # Ctrl+C stops them all

* With Redis (CELERY_BROKER_URL set): N separate worker processes (`--pool=solo`,
  the supported way to run Celery on Windows), one page each at a time.
* Without Redis (default local filesystem broker): ONE worker process with an
  N-thread pool. The filesystem broker claims a message by moving its file, and on
  Windows several consumer processes racing for the same file crash; a single
  consumer with N threads avoids that while still running N pages at once. The
  heavy steps (tesseract subprocesses, PyTorch, OpenCV) release the GIL, so the
  threads really run in parallel. Each page still runs the unchanged pipeline.

Workers load the handwriting model once at start-up. Logs: .celery/logs/.

Production equivalent (Linux, Redis):
    CELERY_BROKER_URL=redis://redis:6379/0 \
    celery -A form_pipeline.tasks worker --pool=prefork --concurrency=4 --prefetch-multiplier=1
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE.parent / ".celery" / "logs"

sys.path.insert(0, str(HERE))
import form_pipeline.config  # noqa: E402,F401 - loads .env (CELERY_BROKER_URL etc.) into the environment


def main():
    cores = os.cpu_count() or 2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--workers", type=int, default=max(1, min(4, cores // 2)),
                    help=f"number of worker processes (default: {max(1, min(4, cores // 2))} on this {cores}-core machine)")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PIPELINE_TORCH_THREADS=str(max(1, cores // args.workers)),
               USE_TF="0", TRANSFORMERS_NO_TF="1", PYTHONUNBUFFERED="1")
    base = [sys.executable, "-m", "celery", "-A", "form_pipeline.tasks", "worker",
            "--loglevel=INFO", "--prefetch-multiplier=1"]
    if os.getenv("CELERY_BROKER_URL"):
        layout = [(f"worker{i}", ["--pool=solo", "-n", f"page-worker{i}@%h"]) for i in range(1, args.workers + 1)]
        desc = f"{args.workers} worker process(es)"
    else:
        layout = [("worker", ["--pool=threads", f"--concurrency={args.workers}", "-n", "page-worker@%h"])]
        desc = f"1 worker process with {args.workers} page slots (local filesystem broker)"
    procs = []
    for name, extra in layout:
        log = open(LOG_DIR / f"{name}.log", "w", encoding="utf-8")
        procs.append((subprocess.Popen(base + extra, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    print(f"Started {desc}; {env['PIPELINE_TORCH_THREADS']} CPU thread(s) per page slot. Logs: {LOG_DIR}\n"
          f"The handwriting model loads (~20-30 s) before pages are taken. Ctrl+C to stop.")
    try:
        while True:
            for p, _ in procs:
                if p.poll() is not None:
                    print(f"worker pid {p.pid} exited with code {p.returncode} - see logs")
                    raise KeyboardInterrupt
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping workers…")
        for p, log in procs:
            p.terminate()
        for p, log in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
            log.close()


if __name__ == "__main__":
    main()
