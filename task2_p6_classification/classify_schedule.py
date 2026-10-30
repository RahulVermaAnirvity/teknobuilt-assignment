"""Classify a Primavera P6 schedule into the Discipline / Sub-Discipline rubric (hybrid RAG).

    python classify_schedule.py ../P6_Schedule_Sample.csv -o ../outputs              # Gemini via .env
    python classify_schedule.py ../P6_Schedule_Sample.csv -o ../outputs --llm none   # no LLM calls

Outputs:
    task2_classified.csv           every row: Discipline / Sub-Discipline / confidence / source /
                                   matched_by (rule id, neighbours or LLM reason) / evidence
    task2_stats.json               how many rows each tier resolved
    task2_llm_batch_requests.jsonl the exact LLM payloads (incl. retrieved evidence), for audit
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from p6classify.pipeline import classify

COLUMNS = ["Activity ID", "Activity Name", "Discipline", "Sub-Discipline", "confidence", "source",
           "matched_by", "needs_review", "evidence", "prefix", "norm"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("-o", "--outdir", type=Path, default=Path("."))
    ap.add_argument("--cache", type=Path, help="JSON {key: label} of human-verified answers")
    ap.add_argument("--llm", choices=["auto", "gemini", "claude", "none"], default="auto",
                    help="LLM provider for the residual activities (auto = Gemini if GEMINI_API_KEY is set)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = pd.read_csv(args.csv, encoding="utf-8-sig", dtype=str).fillna("")
    missing = {"Activity ID", "Activity Name"} - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing column(s): {', '.join(sorted(missing))}")
    cache = json.loads(args.cache.read_text()) if args.cache else {}
    args.outdir.mkdir(parents=True, exist_ok=True)
    out, stats = classify(df, cache, args.llm, requests_out=args.outdir / "task2_llm_batch_requests.jsonl",
                          progress=lambda stage, frac: logging.info("[%3.0f%%] %s", frac * 100, stage))
    stats.pop("details")
    out[COLUMNS].to_csv(args.outdir / "task2_classified.csv", index=False)
    (args.outdir / "task2_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
