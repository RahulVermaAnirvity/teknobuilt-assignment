"""Hybrid retrieval over *labelled examples* (semantic embeddings + BM25, fused with RRF).

Why examples and not the label names: activity names and the rubric share almost no
vocabulary ("Containment - Secondary FA/SEC" vs "Cable Trays Drops"). Measured on the
sample, searching the 41 label names gets 9-21% top-1; searching previously labelled
activities gets ~77% top-1 and ~96% top-5. So retrieval is used to
  (a) auto-label only when the evidence is unanimous and very similar (calibrated), and
  (b) hand the LLM the most similar solved examples + candidate labels (RAG).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import config as C

# Construction jargon that general-purpose embedding models don't know. Expansion is
# appended, not substituted, so BM25 still matches the original token exactly.
ABBREVIATIONS = {
    "vesda": "very early smoke detection fire alarm",
    "npw": "non potable water pipe",
    "ugs": "underground services",
    "hac": "hot aisle containment", "hacs": "hot aisle containment",
    "lthw": "low temperature hot water heating pipework",
    "dhw": "domestic hot water plumbing", "dcw": "domestic cold water plumbing",
    "ahu": "air handling unit hvac", "ahus": "air handling unit hvac",
    "acu": "air conditioning unit hvac", "acus": "air conditioning unit hvac",
    "lv": "low voltage electrical", "mv": "medium voltage electrical",
    "fa": "fire alarm", "bas": "building automation system", "bms": "building management system",
    "ups": "uninterruptible power supply", "mep": "mechanical electrical plumbing",
    "ff&e": "furniture fixtures equipment", "whr": "waste heat recovery",
    "epms": "electrical power monitoring system", "cctv": "security camera",
    "poc": "point of connection", "csa": "civil structural architectural",
}


def expand(norm: str) -> str:
    toks = norm.split()
    extra = [ABBREVIATIONS[t] for t in toks if t in ABBREVIATIONS]
    return " ".join([norm, *extra])


def _tok(s: str) -> list[str]:
    return [t for t in re.split(r"[\s/&,()\-]+", s.lower()) if t]


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(C.EMBED_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)


@dataclass
class Neighbour:
    key: str
    name: str
    prefix: str
    label: str
    similarity: float   # cosine similarity (semantic)
    fused: float        # RRF score used for ranking


class HybridRetriever:
    """In-memory index; a few thousand examples search in milliseconds (no vector DB needed)."""

    def __init__(self, keys: list[str], names: list[str], norms: list[str], prefixes: list[str],
                 labels: list[str], w_sem: float = C.W_SEMANTIC, w_bm25: float = C.W_BM25):
        from rank_bm25 import BM25Okapi
        self.keys, self.names, self.norms, self.prefixes, self.labels = keys, names, norms, prefixes, labels
        self.w_sem, self.w_bm25 = w_sem, w_bm25
        self.docs = [f"{p.lower()} {expand(n)}" for p, n in zip(prefixes, norms)]
        self.E = embed(self.docs)
        self.bm25 = BM25Okapi([_tok(d) for d in self.docs]) if self.docs else None

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def _ranks(scores: np.ndarray) -> np.ndarray:
        order = np.argsort(-scores)
        r = np.empty_like(order)
        r[order] = np.arange(len(scores))
        return r

    def search_many(self, norms: list[str], prefixes: list[str], k: int = C.TOP_K,
                    exclude_norms: list[str] | None = None) -> list[list[Neighbour]]:
        if not len(self) or not norms:
            return [[] for _ in norms]
        qdocs = [f"{p.lower()} {expand(n)}" for p, n in zip(prefixes, norms)]
        Q = embed(qdocs)
        sims = Q @ self.E.T
        out = []
        for i, qd in enumerate(qdocs):
            s_sem = sims[i]
            s_bm = self.bm25.get_scores(_tok(qd))
            fused = (self.w_sem / (C.RRF_K + self._ranks(s_sem))
                     + self.w_bm25 / (C.RRF_K + self._ranks(s_bm)))
            if exclude_norms:  # evaluation: hide the item itself and same-text twins
                fused = np.where(np.asarray(self.norms) == exclude_norms[i], -1.0, fused)
            top = np.argsort(-fused)[:k]
            out.append([Neighbour(self.keys[j], self.names[j], self.prefixes[j], self.labels[j],
                                  round(float(s_sem[j]), 4), round(float(fused[j]), 6))
                        for j in top if fused[j] >= 0])
        return out


def auto_accept(neigh: list[Neighbour], prefix: str) -> str | None:
    """Label from retrieval alone, only under strict, calibrated conditions."""
    top = neigh[:C.AUTO_NEIGHBOURS]
    if len(top) < C.AUTO_NEIGHBOURS:
        return None
    if len({n.label for n in top}) != 1 or any(n.prefix != prefix for n in top):
        return None
    return top[0].label if top[0].similarity >= C.AUTO_ACCEPT_SIM else None


def candidate_labels(neigh: list[Neighbour], limit: int = 5) -> list[str]:
    votes: dict[str, float] = {}
    for rank, n in enumerate(neigh):
        votes[n.label] = votes.get(n.label, 0.0) + 1.0 / (rank + 1)
    return sorted(votes, key=votes.get, reverse=True)[:limit]
