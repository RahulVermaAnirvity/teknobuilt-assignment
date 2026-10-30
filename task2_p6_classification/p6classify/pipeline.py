"""Tiered classification cascade (hybrid RAG).

    rows -> normalise -> dedupe to unique keys
         -> T0 label cache        previously verified answers
         -> T1 keyword rules      free, deterministic, auditable
         -> T2 hybrid retrieval   semantic + BM25 over labelled examples; auto-labels only
                                  when neighbours are unanimous, same prefix, very similar
         -> T3 LLM with RAG       remaining keys + their retrieved evidence, enum-constrained
                                  (worker model -> stronger model for unsure items)
         -> validate against taxonomy -> fan results back out to every row
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from . import config as C
from . import llm, rules
from .normalize import id_prefix, normalize
from .retrieval import HybridRetriever, auto_accept, candidate_labels
from .taxonomy import is_valid, split

LLM_CONF = {"high": 0.9, "medium": 0.75, "low": 0.5}


def make_key(prefix: str, norm: str) -> str:
    return f"{prefix}|{norm}"


def build_units(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prefix"] = df["Activity ID"].map(id_prefix)
    df["norm"] = df["Activity Name"].astype(str).map(normalize)
    df["key"] = [make_key(p, n) for p, n in zip(df["prefix"], df["norm"])]
    return df


def classify(df: pd.DataFrame, cache: dict[str, str] | None = None, llm_provider: str = C.LLM_PROVIDER,
             requests_out: Path | None = None, progress=None) -> tuple[pd.DataFrame, dict]:
    """Returns (rows with labels + provenance, stats). stats["details"] holds per-key evidence
    (neighbours, candidates, LLM decision) for UIs; drop it before serialising stats."""
    say = progress or (lambda stage, frac: None)
    cache = cache or {}
    df = build_units(df)
    units = df.groupby("key").agg(name=("Activity Name", "first"), prefix=("prefix", "first"),
                                  norm=("norm", "first"), rows=("key", "size")).reset_index()
    meta = units.set_index("key")
    res: dict[str, dict] = {}
    details: dict[str, dict] = {}

    # T0 + T1 ---------------------------------------------------------------------------
    say("Rules", 0.05)
    for u in units.itertuples():
        if u.key in cache and is_valid(cache[u.key]):
            res[u.key] = {"label": cache[u.key], "confidence": 1.0, "source": "cache", "by": "verified cache"}
            continue
        r = rules.match(u.norm, u.prefix)
        if r:
            res[u.key] = {"label": r.label, "confidence": r.confidence,
                          "source": "rule" if r.confidence >= rules.ACCEPT else "rule_hint", "by": r.rid}

    # T2: hybrid retrieval over the confidently labelled keys ------------------------------
    say("Building the retrieval index (semantic + BM25)", 0.15)
    bank = [k for k, v in res.items() if v["source"] in ("cache", "rule")]
    pending = [k for k in units["key"] if res.get(k, {}).get("source") in (None, "rule_hint")]
    retriever = HybridRetriever(bank, [meta.at[k, "name"] for k in bank], [meta.at[k, "norm"] for k in bank],
                                [meta.at[k, "prefix"] for k in bank], [res[k]["label"] for k in bank])
    say("Retrieving similar labelled activities", 0.3)
    neigh = dict(zip(pending, retriever.search_many([meta.at[k, "norm"] for k in pending],
                                                    [meta.at[k, "prefix"] for k in pending])))
    for k in pending:
        details[k] = {"neighbours": [n.__dict__ for n in neigh[k]], "candidates": candidate_labels(neigh[k])}
        if k not in res:  # only untouched keys; a rule hint goes to the LLM with its evidence
            lab = auto_accept(neigh[k], meta.at[k, "prefix"])
            if lab:
                res[k] = {"label": lab, "confidence": round(neigh[k][0].similarity, 3), "source": "retrieval",
                          "by": f"{C.AUTO_NEIGHBOURS} unanimous neighbours, e.g. {neigh[k][0].name!r}"}

    # T3: LLM with retrieved evidence ------------------------------------------------------
    llm_keys = [k for k in pending if res.get(k, {}).get("source") in (None, "rule_hint")]
    items = [llm.Item(
        key=k, name=meta.at[k, "name"], prefix=meta.at[k, "prefix"], hint=res.get(k, {}).get("label"),
        examples=[{"name": n.name, "client_prefix": n.prefix, "label": n.label, "similarity": round(n.similarity, 2)}
                  for n in neigh[k]],
        candidates=details[k]["candidates"]) for k in llm_keys]
    if requests_out:  # the exact payloads, for audit / for the Claude Batch API path
        requests_out.write_text("\n".join(json.dumps(llm.ClaudeBatchClassifier.build_request(f"p6-{i:05d}", ch))
                                          for i, ch in enumerate(llm.chunks(items))), encoding="utf-8")
    say(f"Asking the LLM about {len(items)} activities (with retrieved examples)", 0.5)
    decisions, llm_name = ({}, "skipped") if llm_provider == "none" else llm.classify_items(
        items, llm_provider, progress=lambda i, n: say(f"LLM batch {i}/{n}", 0.5 + 0.45 * i / n))
    for k in llm_keys:
        d = decisions.get(k)
        if d is None:
            continue
        details[k]["llm"] = d.__dict__
        if d.label != "UNCERTAIN":  # (code-side check: LLM only returns validated enum labels)
            res[k] = {"label": d.label, "confidence": LLM_CONF[d.confidence], "source": "llm",
                      "by": f"{d.model}: {d.reason}"}
        elif k in res:  # keep the rule hint, still flagged for review
            res[k]["by"] += f" | LLM uncertain: {d.reason}"

    # Fan out to rows ---------------------------------------------------------------------
    def evidence(k):
        ns = details.get(k, {}).get("neighbours", [])[:3]
        return " | ".join(f"{n['name']} [{n['prefix']}] -> {n['label']} ({n['similarity']:.2f})" for n in ns)

    def row(k):
        r = res.get(k)
        if not r or not is_valid(r["label"]):
            why = details.get(k, {}).get("llm", {}).get("reason", "")
            return (None, None, 0.0, "unresolved", f"LLM uncertain: {why}" if why else None, True, evidence(k))
        d, s = split(r["label"])
        review = r["source"] == "rule_hint" or r["confidence"] < rules.ACCEPT
        return (d, s, r["confidence"], r["source"], r["by"], review, evidence(k))

    cols = ["Discipline", "Sub-Discipline", "confidence", "source", "matched_by", "needs_review", "evidence"]
    df[cols] = pd.DataFrame([row(k) for k in df["key"]], index=df.index)
    say("Done", 1.0)
    stats = {
        "rows": int(len(df)),
        "unique_names": int(df["Activity Name"].nunique()),
        "unique_keys_after_normalisation": int(len(units)),
        "dedup_reduction": f"{1 - len(units) / len(df):.1%}",
        "keys_by_source": dict(Counter(res.get(k, {"source": "unresolved"})["source"] for k in units["key"])),
        "rows_by_source": {k: int(v) for k, v in df["source"].value_counts().items()},
        "rows_needing_review": int(df["needs_review"].sum()),
        "retrieval_index_size": len(bank),
        "llm_items_sent": len(items),
        "llm_models": llm_name,
        "details": details,
    }
    return df, stats
