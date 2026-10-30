"""Tier 1 - deterministic, high-precision keyword rules.

Rules are ordered most-specific first; the first match wins. Each carries a
confidence: >= ACCEPT is final, anything lower is a *suggestion* that still goes
to the LLM tier (the suggestion is passed along as a hint).

`prefix` restricts a rule to the client's ID prefix. That is only used for
names that are meaningless without context ("Testing", "Dressing", "Cabling")
- for descriptive names we deliberately ignore the prefix, because the brief
says the client's discipline assignment is often wrong (e.g. ARC rows
containing DHW/DCW plumbing or rack installation).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .taxonomy import label as L

ACCEPT = 0.85


@dataclass(frozen=True)
class Rule:
    rid: str
    pattern: re.Pattern
    label: str
    confidence: float
    prefix: Optional[frozenset] = None


def R(rid, rx, disc, sub, conf=0.9, prefix=None) -> Rule:
    return Rule(rid, re.compile(rx, re.I), L(disc, sub), conf, frozenset(prefix) if prefix else None)


RULES: list[Rule] = [
    # ---- Fire protection / fire alarm ---------------------------------------------------
    R("fp.sprinkler", r"sprinkler", "Fire Protection", "Installation", 0.97),
    R("fa.vesda", r"\bvesda\b", "Fire Alarm", "Pull, Test and Terminate", 0.92),
    # ---- Security ------------------------------------------------------------------------
    R("sec.secondary_containment", r"secondary containment for (access control|alarms|video)",
      "Electrical", "Cable Trays Drops", 0.8),  # containment is electrical scope, but SEC-owned -> verify
    R("sec.systems", r"access control|video network|cctv|intruder|\balarms? (cables|devices)\b",
      "Security", "Pull, Test and Terminate", 0.95),
    # ---- Telecom -------------------------------------------------------------------------
    R("tel.runner", r"fiber runner|mount brackets", "Telecommunications", "Basket Tray Main", 0.86, prefix={"TELE"}),
    R("tel.splice", r"\bsplice|fiber|patching|\bdressing\b", "Telecommunications", "Production Fiber", 0.92,
      prefix={"TELE"}),
    R("tel.fiber_generic", r"^(cable pulls|cabling|terminations|testing|labeling)$",
      "Telecommunications", "Production Fiber", 0.86, prefix={"TELE"}),
    R("tel.radio", r"wifi|\bdas\b|radio|antenna", "Telecommunications", "Digital Antenna System / Radio Over IP", 0.85),
    R("tel.misc", r"phone install|install switches|idfs|mors|level and adjust|final check clean down",
      "Telecommunications", "Miscellaneous", 0.85, prefix={"TELE"}),
    R("tel.rack", r"rack installation", "Telecommunications", "Miscellaneous", 0.75),
    # ---- Electrical (incl. mixed low-voltage system containment owned by ELE) ------------
    R("ele.mv_pull", r"(mv|lv|mv/lv)\s*cable pulling|installation of mv/lv cabling|^cabling (lv|main lv)|cabling main lv",
      "Electrical", "Cable Pulling", 0.95),
    R("ele.mv_term", r"termination of mv/lv|mv cable termination|^terminations lv$|terminations main equipment",
      "Electrical", "Cable Termination", 0.93),
    R("ele.secondary_cont", r"secondary containment|containment secondary|containment.*drops|2nd fix containment",
      "Electrical", "Cable Trays Drops", 0.88),
    R("ele.primary_cont", r"primary containment|containment primary|main tray|1st fix containment|containment for lv|containment.*basket",
      "Electrical", "Cable Trays Main", 0.9),
    R("ele.elec_cabling", r"\belec\b.*cabling|^elec cabling$", "Electrical", "Cable Pulling", 0.9),
    R("ele.elec_term", r"\belec\b.*terminations", "Electrical", "Cable Termination", 0.9),
    R("ele.earthing", r"earth(ing)? bars?|earthing|bonding", "Electrical", "Rough-In", 0.85),
    R("ele.tapoff", r"tap off box|connection of electrical supplies|equipment installation cfci",
      "Electrical", "Miscellaneous", 0.85),
    # Mixed FA/Security/BAS low-voltage cabling under ELE: genuinely ambiguous -> hint only, LLM decides
    R("ele.mixed_lv_cabling", r"cabling.*\b(fa|gs|bms|bas)\b|secondary cabling", "Electrical", "Cable Pulling", 0.7),
    R("ele.mixed_lv_term", r"terminations.*\b(fa|gs|bms|bas)\b|secondary terminations", "Electrical", "Cable Termination", 0.7),
    R("ele.testing", r"testing.*mv/lv|mv/lv.*testing|^mep testing", "Electrical", "Miscellaneous", 0.75, prefix={"ELE"}),
    R("ele.pv", r"\bpv panels?\b", "Electrical", "Miscellaneous", 0.8),
    # ---- Plumbing ------------------------------------------------------------------------
    R("plb.water", r"\bdhw\b|\bdcw\b|sanitaryware|vanity|restroom|soils? & waste|npw (1st|2nd) fix|condensate drain",
      "Plumbing", "Rough-In", 0.9),
    # ---- Location-independent architectural items that mention MEP words (dampers) ----
    R("civ.pilecaps", r"pile caps", "Civil", "Foundation Works", 0.9),
    R("arc.ceiling", r"ceiling (grid|tiles|hangers|panels|system)|close ceilings|eurobond ceiling|ceiling in central corridor|hangers for the ceiling|openings in ceiling",
      "Architectural", "Acoustical Ceiling Tile", 0.9),
    R("arc.exterior", r"flashings?|louvers?|snow hoods|sky ?lights|roof walkways|weathering of plantroom",
      "Architectural", "Exterior Finishes", 0.87),
    # ---- Mechanical ----------------------------------------------------------------------
    R("mec.modules", r"plantroom modules|pump house modules|across module divides|tie in to ahu",
      "Mechanical", "Exterior Installation", 0.85),
    R("mec.hvac", r"duct|\bahus?\b|\bacus?\b|\bac (units|pipework|support)|dampers|transfer fans|refrigeration|lthw|leak detection|chimney|test & balancing|pressure test|pipework (drops|connections)|pipe drops|first fix pipework",
      "Mechanical", "Rough-In", 0.88),
    R("mec.handrails", r"mechanical corridor handrails", "Structural", "Miscellaneous", 0.75),
    # ---- Architectural -------------------------------------------------------------------
    R("arc.hac", r"\bhacs?\b|hot aisle", "Architectural", "Hot Aisle Containment", 0.93),
    R("arc.interior", r"paint(ing)?\b(?!.*floor markings)|plasterboard|partitions|doors|ironmongery|floor finishes|tile walls|skim|tape & joint|white walls|glazing|column encasements?|ff&e|furniture|access hatches|mesh walls|corridor & riser walls|roof (fire )?lining|decoration|floor markings|insulation & labeling",
      "Architectural", "Interior Finishes", 0.88),
    R("arc.firestop", r"firestop", "Architectural", "Interior Finishes", 0.8),
    R("arc.misc", r"lift (installation|commissioning)|snagging|qa walkdown|walkdowns", "Architectural", "Miscellaneous", 0.85),
    # ---- Structural ----------------------------------------------------------------------
    R("str.envelope", r"cladding|curtain walling|sandwich panel|roof decking|roof build up|building envelope|^roof$|roofing|^panels$",
      "Structural", "Envelope", 0.88),
    R("str.equipment_steel", r"generator (support|access) steel|whr supports|flue towers|panel support structures",
      "Structural", "Exterior Equipment Steel", 0.85),
    R("str.frame", r"steel frame|structural steel|penthouses|gantry steel|secondary steel|steelwork supports|staircase",
      "Structural", "Miscellaneous", 0.86),
    R("str.elevated", r"^decking$|level \d.*(slab|curing)|concrete cores|wall pours|roof pour|loading bay installation",
      "Structural", "Miscellaneous", 0.75),
    # ---- Civil ---------------------------------------------------------------------------
    R("civ.pads", r"fill & detailing for pads|concrete pads|\bpads?\b.*(prep|pour)|loading dock slab|bin store & compactor slab|tx/lbp|gpc|\bums\d*\b.*(slab|curing)",
      "Civil", "Exterior Equipment Pad", 0.86),
    R("civ.foundation", r"pil(e|es|ing)\b|pile caps|ground beams|foundation|sheet piling|anchors|rock bolts|waler beam|columns|preloading|surcharge|overload settlement|waiting time for anchors|retaining wall|msstb",
      "Civil", "Foundation Works", 0.88),
    R("civ.slab", r"floor (prep|pour)|slab (prep|pour|curing)|yard slab|slab curing|floor build up|^slab|micro pile slab|curing period",
      "Civil", "Slab on Grade", 0.86),
    R("civ.wet", r"foul water|storm water|rain ?water|potable water|\bnpw\b|sewer|hydrant|drainage|discharge|attenuation|tank|aco channels|sub drainage|district heating|waste heat recovery|radon header",
      "Civil", "Underground Wet Utilities", 0.86),
    R("civ.dry", r"ugs install (lv|mv)|^(lv|mv)$|street lighting|^telecoms?$|services installation|directional drilling",
      "Civil", "Underground Dry Utilities", 0.86),
    R("civ.misc", r"site clearance|earthworks|access routes|excavation|tree felling|rock blasting|top soil|mass exchange|access roads?|road (build up|surfacing)|link road|compound|site (set up|survey)|water management|handover|back ?fill|subsoil|enabling work|drilling holes",
      "Civil", "Miscellaneous", 0.86),
    R("lan.landscape", r"landscaping|green roof planting|planting", "Landscape", "Miscellaneous", 0.92),
]


def match(norm_name: str, prefix: str) -> Optional[Rule]:
    for rule in RULES:
        if rule.prefix and prefix not in rule.prefix:
            continue
        if rule.pattern.search(norm_name):
            return rule
    return None
