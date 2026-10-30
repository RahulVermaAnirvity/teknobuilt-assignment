"""Measure the retrieval and RAG tiers instead of assuming they work.

    python evaluate.py ../P6_Schedule_Sample.csv                    # retrieval only (free)
    python evaluate.py ../P6_Schedule_Sample.csv --llm-sample 40    # + live Gemini RAG test

Ground truth: there is no labelled answer key in the brief, so the activities labelled by the
high-precision keyword rules serve as pseudo-gold. That is fine for *comparing* methods and
calibrating thresholds; a human-labelled gold set is the production requirement.

Leakage control: when an activity is evaluated, every example with the same normalised name
is hidden from the index (no self-lookup via a twin under another prefix).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from p6classify import config as C
from p6classify import llm, rules
from p6classify.pipeline import build_units
from p6classify.retrieval import HybridRetriever, candidate_labels


def load_gold(csv: Path) -> pd.DataFrame:
    df = build_units(pd.read_csv(csv, encoding="utf-8-sig", dtype=str).fillna(""))
    units = df.groupby("key").agg(name=("Activity Name", "first"), prefix=("prefix", "first"),
                                  norm=("norm", "first")).reset_index()
    labels = []
    for u in units.itertuples():
        r = rules.match(u.norm, u.prefix)
        labels.append(r.label if r and r.confidence >= rules.ACCEPT else None)
    units["label"] = labels
    return units.dropna(subset=["label"]).reset_index(drop=True)


def retrieval_eval(gold: pd.DataFrame, w_sem: float, w_bm25: float):
    r = HybridRetriever(gold.key.tolist(), gold.name.tolist(), gold.norm.tolist(), gold.prefix.tolist(),
                        gold.label.tolist(), w_sem=w_sem, w_bm25=w_bm25)
    neigh = r.search_many(gold.norm.tolist(), gold.prefix.tolist(), k=C.TOP_K, exclude_norms=gold.norm.tolist())
    truth = gold.label.tolist()
    top1 = np.mean([bool(n) and candidate_labels(n)[0] == t for n, t in zip(neigh, truth)])
    top5 = np.mean([t in candidate_labels(n, 5) for n, t in zip(neigh, truth)])
    return top1, top5, neigh


def auto_accept_curve(gold: pd.DataFrame, neigh) -> list[dict]:
    rows = []
    for thr in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        accepted = correct = 0
        for n, p, t in zip(neigh, gold.prefix, gold.label):
            top = n[:C.AUTO_NEIGHBOURS]
            if (len(top) == C.AUTO_NEIGHBOURS and len({x.label for x in top}) == 1
                    and all(x.prefix == p for x in top) and top[0].similarity >= thr):
                accepted += 1
                correct += top[0].label == t
        rows.append({"threshold": thr, "auto_accepted": accepted, "coverage": round(accepted / len(gold), 3),
                     "precision": round(correct / accepted, 3) if accepted else None})
    return rows


def llm_eval(gold: pd.DataFrame, n: int, seed: int = 7) -> dict:
    """Hold out n activities (stratified by label), hide them and their same-name twins from the
    index, and let the live RAG tier classify them from evidence alone."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for i, lab in enumerate(gold.label):
        by_label[lab].append(i)
    picks = []
    while len(picks) < min(n, len(gold)):
        for lab in sorted(by_label):
            if by_label[lab] and len(picks) < n:
                picks.append(by_label[lab].pop(rng.randrange(len(by_label[lab]))))
    test = gold.iloc[picks]
    bank = gold[~gold.norm.isin(set(test.norm))]
    r = HybridRetriever(bank.key.tolist(), bank.name.tolist(), bank.norm.tolist(), bank.prefix.tolist(),
                        bank.label.tolist())
    neigh = r.search_many(test.norm.tolist(), test.prefix.tolist())
    items = [llm.Item(k, nm, p, None,
                      [{"name": x.name, "client_prefix": x.prefix, "label": x.label, "similarity": round(x.similarity, 2)}
                       for x in ns], candidate_labels(ns))
             for k, nm, p, ns in zip(test.key, test.name, test.prefix, neigh)]
    decisions, models = llm.classify_items(items)
    answered = [(k, t) for k, t in zip(test.key, test.label) if k in decisions]
    labelled = [(k, t) for k, t in answered if decisions[k].label != "UNCERTAIN"]
    high = [(k, t) for k, t in labelled if decisions[k].confidence == "high"]
    retr_top1 = np.mean([bool(ns) and candidate_labels(ns)[0] == t for ns, t in zip(neigh, test.label)])
    wrong = [{"activity": test.set_index("key").at[k, "name"], "expected": t, "llm": decisions[k].label,
              "confidence": decisions[k].confidence, "reason": decisions[k].reason}
             for k, t in labelled if decisions[k].label != t]
    return {
        "held_out": len(test), "models": models, "answered": len(answered),
        "uncertain": sum(decisions[k].label == "UNCERTAIN" for k, _ in answered),
        "accuracy_when_labelled": round(np.mean([decisions[k].label == t for k, t in labelled]), 3) if labelled else None,
        "high_confidence": len(high),
        "precision_high_confidence": round(np.mean([decisions[k].label == t for k, t in high]), 3) if high else None,
        "retrieval_top1_same_items": round(float(retr_top1), 3),
        "errors": wrong,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--llm-sample", type=int, default=0, help="also test the live RAG tier on N held-out activities")
    ap.add_argument("-o", "--output", type=Path, default=Path("../outputs/task2_eval.json"))
    args = ap.parse_args()
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

    gold = load_gold(args.csv)
    print(f"pseudo-gold: {len(gold)} rule-labelled activities, {gold.label.nunique()} distinct labels\n")

    weights = [(1.0, 0.0), (1.0, 0.3), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]
    table, best = [], None
    for w_sem, w_bm in weights:
        top1, top5, neigh = retrieval_eval(gold, w_sem, w_bm)
        name = {(1.0, 0.0): "semantic only", (0.0, 1.0): "BM25 only"}.get((w_sem, w_bm), f"hybrid {w_sem:g}:{w_bm:g}")
        table.append({"method": name, "w_semantic": w_sem, "w_bm25": w_bm,
                      "top1": round(float(top1), 3), "top5_recall": round(float(top5), 3)})
        if (w_sem, w_bm) == (C.W_SEMANTIC, C.W_BM25):
            best = neigh
    print(pd.DataFrame(table).to_string(index=False))
    if best is None:
        _, _, best = retrieval_eval(gold, C.W_SEMANTIC, C.W_BM25)

    curve = auto_accept_curve(gold, best)
    print(f"\nauto-accept (configured weights {C.W_SEMANTIC:g}:{C.W_BM25:g}; top-{C.AUTO_NEIGHBOURS} unanimous + same prefix):")
    print(pd.DataFrame(curve).to_string(index=False))

    result = {"pseudo_gold": len(gold), "retrieval": table, "auto_accept_curve": curve,
              "configured": {"w_semantic": C.W_SEMANTIC, "w_bm25": C.W_BM25, "auto_accept_sim": C.AUTO_ACCEPT_SIM}}
    if args.llm_sample:
        print(f"\nLive RAG test on {args.llm_sample} held-out activities…")
        result["llm"] = llm_eval(gold, args.llm_sample)
        summary = {k: v for k, v in result["llm"].items() if k != "errors"}
        print(json.dumps(summary, indent=2))
        for e in result["llm"]["errors"]:
            print(f"  ✗ {e['activity']!r}: expected {e['expected']}, got {e['llm']} ({e['confidence']}) - {e['reason']}")
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
