"""Canonicalise activity names so identical *work* collapses to one cache key.

"Floor Pour - COLO3  Cell 3" and "Floor Pour - COLO1  Cell 1" are the same
classification problem; only the location differs. Stripping location,
sequencing and phase tokens is what turns ~4.5k rows into a few hundred
unique questions - the single biggest cost lever in the design.
"""
from __future__ import annotations

import re

TYPOS = {
    "containmnet": "containment", "temination": "termination", "secuirty": "security",
    "copmound": "compound", "conections": "connections", "condenstate": "condensate",
    "pipwork": "pipework", "chimey": "chimney", "caballing": "cabling", "chanels": "channels",
    "inital": "initial", "lokers": "lockers", "fibre": "fiber", "labelling": "labeling",
    "transfo": "transformer", "trafo": "transformer",
}

_LOCATION_PATTERNS = [
    r"\bhel1[67]\b", r"\bcolo\s*\d(?:\s*&\s*\d)?\b", r"\bcell\s*\d(?:\s*&\s*\d)?\b",
    r"\bce\d(?:\s*&\s*\d)?\b", r"\bgl\s*:?\s*[a-z]?\d+(?:\s*-\s*(?:gl\s*)?[a-z]?\d+)?\b",
    r"\(gl:[^)]*\)", r"\bzone\s+[a-f]\d?\b", 
    r"\bmmr\s*\d\b", r"\b(?:part\s*)?\d\s*/\s*\d\b", r"\(\s*\)", r"\bc[1-5]_sub\b",
    r"\b(?:east|west|north|south)(?: side)?\b", r"\bfrom\b", r"\bsecure store i{1,2}\b",
    r"\b(?:fcr|foc|idf|ser|soc|high value store)\b", r"\b(?:admin|admin center)\b",
    r"\bpoles?\s*\d+\s*-\s*\d+\b", r"\(\d+m3\)", r"\b\d+x\b", r"\bsouthern parcel\b",
    r"#\s*\d+", r"\bzone f peat\b", 
]
_LOC_RE = re.compile("|".join(_LOCATION_PATTERNS), re.I)


def normalize(name: str) -> str:
    s = name.lower()
    for bad, good in TYPOS.items():
        s = re.sub(rf"\b{bad}\b", good, s)
    s = _LOC_RE.sub(" ", s)
    s = re.sub(r"[^a-z0-9&/+ ]", " ", s)
    return re.sub(r"\s+", " ", s).strip() or name.lower().strip()


def id_prefix(activity_id: str) -> str:
    m = re.match(r"^([A-Z]+)", activity_id or "")
    return m.group(1) if m else ""
