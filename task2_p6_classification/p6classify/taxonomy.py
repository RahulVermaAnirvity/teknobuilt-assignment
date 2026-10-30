"""The closed taxonomy - the single source of truth for every tier.

Labels are (discipline, sub_discipline) PAIRS, never two independent fields:
that makes "Plumbing / Cable Pulling" unrepresentable rather than merely
unlikely. Every tier (rules, kNN, LLM) emits a label id from LABELS and every
result is re-checked against it before it is written out.
"""
from __future__ import annotations

TAXONOMY: dict[str, list[str]] = {
    "Architectural": ["Acoustical Ceiling Tile", "Exterior Finishes", "Hot Aisle Containment",
                      "Interior Finishes", "Miscellaneous"],
    "Building Automation System": ["Pull, Test and Terminate"],
    "Civil": ["Exterior Equipment Pad", "Foundation Works", "Miscellaneous", "Slab on Grade",
              "Underground Dry Utilities", "Underground Wet Utilities"],
    "Electrical": ["Cable Pulling", "Cable Termination", "Cable Trays Drops", "Cable Trays Main",
                   "Miscellaneous", "Rough-In"],
    "Fire Alarm": ["Pull, Test and Terminate"],
    "Fire Protection": ["Installation"],
    "Landscape": ["Miscellaneous"],
    "Mechanical": ["Exterior Installation", "Rough-In"],
    "Plumbing": ["Rough-In"],
    "Security": ["Pull, Test and Terminate"],
    "Structural": ["Envelope", "Exterior Equipment Steel", "Miscellaneous"],
    "Telecommunications": ["Basket Tray Main", "Digital Antenna System / Radio Over IP",
                           "Miscellaneous", "Production Fiber"],
}

SEP = " :: "
LABELS: list[str] = [f"{d}{SEP}{s}" for d, subs in TAXONOMY.items() for s in subs]
LABEL_SET = frozenset(LABELS)

# Client ID prefix -> the discipline the client *claims*. Used only as a weak
# prior / tie-breaker, because the brief says these are often wrong.
PREFIX_HINT = {
    "ARC": "Architectural", "CON": "Civil", "ELE": "Electrical", "MEC": "Mechanical",
    "ME": "Mechanical", "SEC": "Security", "TELE": "Telecommunications", "GEN": None,
}


def label(discipline: str, sub: str) -> str:
    lab = f"{discipline}{SEP}{sub}"
    if lab not in LABEL_SET:  # guards hand-written rules against typos too
        raise ValueError(f"not in taxonomy: {lab!r}")
    return lab


def split(lab: str) -> tuple[str, str]:
    d, s = lab.split(SEP, 1)
    return d, s


def is_valid(lab: str | None) -> bool:
    return lab in LABEL_SET
