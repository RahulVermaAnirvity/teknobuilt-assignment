"""Teknobuilt · Construction Data Automation - demo console for both assessment tasks.

    streamlit run app.py

Tabs:
  Digitize forms   - run the Task 1 pipeline on the sample PDF or an upload, inspect every field
  Review queue     - correct flagged fields (human-in-the-loop) and export corrected JSON
  Robustness lab   - degrade a page (skew, mud, blur, crossed-out ...) and compare with the clean read
  P6 classifier    - run the Task 2 cascade on the sample schedule or an uploaded CSV
  How it works     - architecture summary
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "task1_form_digitization"), str(ROOT / "task2_p6_classification")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

ASSETS = ROOT / "app_assets"
SAMPLE_PDF = ROOT / "Sample_input.pdf"
SAMPLE_CSV = ROOT / "P6_Schedule_Sample.csv"
NAVY, ORANGE = "#043448", "#F08404"

st.set_page_config(page_title="Teknobuilt · Construction Data Automation",
                   page_icon=str(ASSETS / "teknobuilt_mark.png"), layout="wide")

from form_pipeline import config as C  # noqa: E402
from form_pipeline.extract import process_page  # noqa: E402
from form_pipeline.ingest import RawPage, iter_pages  # noqa: E402
from form_pipeline.ocr import handwriting_engine, tesseract_available  # noqa: E402
from form_pipeline.vlm import make_vlm  # noqa: E402
from p6classify import config as P6  # noqa: E402
from p6classify import llm as p6llm, rules  # noqa: E402
from p6classify.normalize import normalize  # noqa: E402
from p6classify.pipeline import classify  # noqa: E402
from p6classify.retrieval import HybridRetriever, auto_accept, candidate_labels  # noqa: E402
from run_pipeline import celery_workers_online, run as run_forms, run_celery  # noqa: E402
import stress_test as degrade  # noqa: E402


# ============================================================================ branding

def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.2rem; }}
  .tb-header {{ display:flex; align-items:center; gap:28px; padding:18px 4px 14px 4px; flex-wrap:wrap; }}
  .tb-header img {{ height:38px; }}
  .tb-divider {{ width:1px; height:40px; background:#D5DEE3; }}
  .tb-eyebrow {{ font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; color:{ORANGE}; font-weight:700; }}
  .tb-title {{ font-size:1.55rem; font-weight:700; color:{NAVY}; line-height:1.2; }}
  .tb-accent {{ height:5px; border-radius:3px; margin:0 0 18px 0;
               background:linear-gradient(90deg,{NAVY} 0%,{NAVY} 70%,{ORANGE} 70%,{ORANGE} 100%); }}
  .tb-badge {{ display:inline-block; padding:3px 12px; border-radius:999px; font-weight:600; font-size:.82rem; }}
  .tb-card {{ border:1px solid #E1E8EC; border-left:5px solid {ORANGE}; border-radius:8px;
             padding:12px 16px; background:#FFFFFF; }}
  .tb-card .lbl {{ font-size:.76rem; color:#5B6B73; text-transform:uppercase; letter-spacing:.06em; }}
  .tb-card .val {{ font-size:1.6rem; font-weight:700; color:{NAVY}; }}
  .tb-muted {{ color:#5B6B73; font-size:.88rem; }}
  .tb-footer {{ margin-top:36px; padding-top:12px; border-top:1px solid #E1E8EC; color:#7A8A92; font-size:.8rem; }}
  section[data-testid="stSidebar"] {{ border-right:3px solid {NAVY}; }}
  div[data-testid="stTabs"] button[aria-selected="true"] p {{ color:{NAVY}; font-weight:700; }}
</style>
<div class="tb-header">
  <img src="data:image/png;base64,{_b64(ASSETS / 'teknobuilt_wordmark.png')}" alt="Teknobuilt"/>
  <div class="tb-divider"></div>
  <div><div class="tb-eyebrow">AI-ML Engineering Assessment</div>
       <div class="tb-title">Construction Data Automation Console</div></div>
</div>
<div class="tb-accent"></div>
""", unsafe_allow_html=True)

STATUS = {
    "ok": ("#E7F6EF", "#1B7F5A", "Accepted"),
    "needs_review": ("#FFF1E0", "#B45F00", "Needs review"),
    "rejected": ("#FDECEC", "#B42318", "Rejected"),
    "error": ("#EEF0F2", "#475467", "Error"),
}
ANSWER_LABEL = {"S": "S · Safe", "R": "R · At risk", "I": "I · Imminent hazard", "N": "N · N/A"}
TEXT_FIELDS = [("observation_date", "Observation date"), ("observer", "Observer"),
               ("company_observed", "Company observed"), ("timesheet_number", "Timesheet #"),
               ("employee_number", "Employee #"), ("checklist_title", "Form type"), ("comments", "Comments")]


def badge(status: str) -> str:
    bg, fg, label = STATUS[status]
    return f'<span class="tb-badge" style="background:{bg};color:{fg}">{label}</span>'


def card(label: str, value) -> str:
    return f'<div class="tb-card"><div class="lbl">{label}</div><div class="val">{value}</div></div>'


def cards(items: list[tuple[str, object]]):
    for col, (label, value) in zip(st.columns(len(items)), items):
        col.markdown(card(label, value), unsafe_allow_html=True)


def short_engine(engine: str | None) -> str:
    if not engine:
        return "—"
    if engine.startswith("trocr"):
        return "TrOCR (local)"
    if engine.startswith("gemini:"):
        return f"Gemini · {engine.split(':', 1)[1]}"
    if engine.startswith("claude:"):
        return f"Claude · {engine.split(':', 1)[1]}"
    return engine.capitalize()


def thumb(gray: np.ndarray, width: int = 760) -> np.ndarray:
    h, w = gray.shape[:2]
    return cv2.resize(gray, (width, int(h * width / w)), interpolation=cv2.INTER_AREA) if w > width else gray


# ============================================================================ sidebar

gemini_key = bool(os.getenv("GEMINI_API_KEY"))


@st.cache_data(ttl=30, show_spinner=False)
def _workers_online() -> int:
    """Pages Celery can process at once (sum of worker pool sizes), cached briefly. The
    local filesystem broker polls about once a second, so the query needs a ~2.5 s window."""
    return celery_workers_online(timeout=2.5)


with st.sidebar:
    st.image(str(ASSETS / "teknobuilt_mark.png"), width=56)
    st.markdown(f"<h3 style='color:{NAVY};margin-top:4px'>Pipeline settings</h3>", unsafe_allow_html=True)
    use_vlm = st.toggle("Escalate unclear fields to Gemini", value=gemini_key, disabled=not gemini_key,
                        help="Only handwritten fields that fail the local confidence/validation gate are sent, "
                             "as small crops. The VLM is a second reader: its answer is accepted only if it "
                             "agrees with the local model (or the local reading was invalid and the VLM is "
                             "highly confident).")
    st.caption("Vision model key: **detected** (from `.env`)" if gemini_key
               else "Vision model key: **not set** - add `GEMINI_API_KEY` to `.env` to enable.")
    st.divider()
    st.markdown("**Execution**")
    workers_online = _workers_online()
    use_celery = st.toggle(f"Parallel pages on Celery ({workers_online} pages at a time)", value=workers_online > 0,
                           disabled=workers_online == 0,
                           help="Each page becomes one Celery task, so several pages are processed at the same "
                                "time. Workers run the identical per-page pipeline - output is the same as a "
                                "local run, only faster.")
    if not workers_online:
        st.caption("No workers running. Start them with:\n`python task1_form_digitization/start_workers.py -n 4`")
    st.divider()
    st.markdown("**Engines**")
    st.caption(f"Printed text: Tesseract {'✓' if tesseract_available() else '✗ not found'}")
    st.caption(f"Handwriting: `{C.HANDWRITING_MODEL}`")
    st.caption(f"VLM models: {', '.join(C.GEMINI_MODELS)}")
    st.caption("Checkboxes: ink-density measurement (no AI)")
    st.divider()
    st.markdown("**Acceptance gates**")
    st.caption(f"Handwriting accept ≥ {C.FIELD_ACCEPT_CONF:.2f} · checkbox winner ≥ {C.CHECKBOX_RELATIVE}× runner-up")
    st.caption(f"Blur warning < {C.BLUR_LAPLACIAN_VAR:.0f} · blank page < {C.BLANK_INK_RATIO:.1%} ink")


@st.cache_resource(show_spinner="Loading the handwriting model (first run only)…")
def warm_models():
    handwriting_engine().available  # noqa: B018 - triggers the lazy load once per server
    return True


@st.cache_resource
def sample_pages() -> list[RawPage]:
    return list(iter_pages(SAMPLE_PDF))


tab_run, tab_review, tab_lab, tab_p6, tab_about = st.tabs([
    ":material/document_scanner: Digitize forms", ":material/fact_check: Review queue",
    ":material/science: Robustness lab", ":material/account_tree: P6 classifier",
    ":material/info: How it works"])


# ============================================================================ tab 1: digitize

def render_page(res: dict, image: np.ndarray | None):
    head = f"Page {res['page_number']} &nbsp; {badge(res['status'])}"
    if res.get("rejection_reason"):
        head += f" &nbsp;<span class='tb-muted'>{res['rejection_reason']}</span>"
    head += f" &nbsp;<span class='tb-muted'>· {res['processing_ms'] / 1000:.1f}s</span>"
    st.markdown(head, unsafe_allow_html=True)

    left, right = st.columns([2, 3], gap="large")
    with left:
        if image is not None:
            st.image(thumb(image), caption=f"Page {res['page_number']} (as scanned)")
        q = res["quality"]
        st.caption(f"Ink {q['ink_ratio']:.1%} · blur score {q['blur_score']:.0f} · deskew {q['deskew_angle_deg']}° · "
                   f"rotation {q['rotation_applied_deg']}° · grid {q['grid_h_lines']}×{q['grid_v_lines']} lines")
        if q["warnings"]:
            st.warning("Quality warnings: " + ", ".join(q["warnings"]))
    with right:
        form = res.get("form")
        if not form:
            st.info("No data extracted from this page - it was refused by the quality gate, "
                    "so nothing can flow downstream from it.")
            return
        rows = [{"Field": label, "Value": form[k]["value"] or "—", "Confidence": form[k]["confidence"],
                 "Status": "Review" if form[k]["needs_review"] else "OK", "Read by": short_engine(form[k]["engine"]),
                 "Flags": ", ".join(form[k]["flags"]), "Alternatives": ", ".join(form[k]["alternatives"])}
                for k, label in TEXT_FIELDS]
        for k, label in (("location_cwa", "Location (CWA)"), ("shift", "Shift")):
            f = form[k]
            rows.insert(5, {"Field": label, "Value": (f["value"] or "—").title() if k == "shift" else f["value"] or "—",
                            "Confidence": f["confidence"], "Status": "Review" if f["needs_review"] else "OK",
                            "Read by": "Ink density", "Flags": ", ".join(f["flags"]), "Alternatives": ""})
        st.markdown("**Header & handwritten fields**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Confidence": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})

        st.markdown(f"**Checklist - {form['checklist_title']['value'] or 'unknown form'}**")
        st.dataframe(pd.DataFrame([{
            "#": it["item_no"], "Question": it["question"]["value"],
            "Answer": ANSWER_LABEL.get(it["response"]["value"], "—"),
            "Confidence": it["response"]["confidence"],
            "Status": "Review" if it["response"]["needs_review"] else "OK",
            "Flags": ", ".join(it["response"]["flags"])} for it in form["checklist"]]),
            hide_index=True, width="stretch", column_config={
                "Confidence": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})

        st.markdown("**CCZJV expectation questions**")
        st.dataframe(pd.DataFrame([{
            "Question": it["question"], "Answer": (it["response"]["value"] or "—").upper(),
            "Status": "Review" if it["response"]["needs_review"] else "OK",
            "Flags": ", ".join(it["response"]["flags"])} for it in form["expectation_questions"]]),
            hide_index=True, width="stretch")
    with st.expander("Raw JSON for this page"):
        st.json(res, expanded=False)


with tab_run:
    st.markdown("Upload scanned safety observation forms (PDF or image). Each page goes through the "
                "quality gate, layout detection, checkbox measurement, OCR/handwriting recognition and "
                "validation; unclear fields are escalated or flagged - never guessed.")
    c1, c2 = st.columns([1, 2])
    source = c1.radio("Input", ["Sample PDF (provided)", "Upload file"] if SAMPLE_PDF.exists() else ["Upload file"],
                      label_visibility="collapsed")
    upload = None
    if source == "Upload file":
        upload = c2.file_uploader("PDF or image", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
    else:
        c2.caption(f"`{SAMPLE_PDF.name}` - 3 handwritten forms + 1 explicitly failed test form.")

    if st.button("Run pipeline", type="primary", disabled=(source == "Upload file" and upload is None)):
        if not use_celery:
            warm_models()  # Celery workers keep their own copy of the model loaded
        if upload is not None:
            # UPLOAD_DIR is a volume shared with the Celery workers (docker compose); otherwise a temp dir
            base = Path(os.getenv("UPLOAD_DIR") or tempfile.mkdtemp())
            base.mkdir(parents=True, exist_ok=True)
            tmp = base / f"{os.urandom(4).hex()}_{Path(upload.name).name}"
            tmp.write_bytes(upload.getvalue())
            path = tmp
        else:
            path = SAMPLE_PDF
        try:
            import pymupdf
            total = pymupdf.open(path).page_count if path.suffix.lower() == ".pdf" else 1
        except Exception:
            total = 1
        bar = st.progress(0.0, text="Starting…")
        images: dict[int, np.ndarray] = {}
        done: list[int] = []

        def on_page(raw, res):
            done.append(res.page_number)
            if raw is not None and raw.image is not None:
                images[raw.page_number] = thumb(raw.image)
            bar.progress(min(len(done) / max(total, 1), 1.0),
                         text=f"{len(done)}/{total} pages done · page {res.page_number}: {STATUS[res.status][2]}")

        mode = f"Celery, {workers_online} pages at a time" if use_celery else "this process, one page at a time"
        with st.spinner(f"Processing pages ({mode})…"):
            out = (run_celery if use_celery else run_forms)(path, use_vlm, on_page=on_page)
        bar.empty()
        st.session_state["t1"] = {"output": out.model_dump(mode="json"), "images": images}
        st.session_state.pop("corrections", None)

    t1 = st.session_state.get("t1")
    if t1:
        out = t1["output"]
        s = out["summary"]
        avg = np.mean([p["processing_ms"] for p in out["pages"] if p["form"]] or [0]) / 1000
        cards([("Pages", s["pages_total"]), ("Accepted", s["ok"]), ("Needs review", s["needs_review"]),
               ("Rejected / error", s["rejected"] + s["error"]), ("Avg s / form page", f"{avg:.1f}")])
        st.caption(f"Engines - handwriting: {short_engine(out['engines']['handwriting'])} · "
                   f"escalation: {out['engines']['vlm_escalation']} · execution: "
                   f"{out['engines'].get('execution', 'local')} · run at {out['processed_at']}")
        st.download_button("Download JSON", json.dumps(out, indent=2), file_name="forms_output.json",
                           mime="application/json")
        for res in out["pages"]:
            with st.container(border=True):
                render_page(res, t1["images"].get(res["page_number"]))


# ============================================================================ tab 2: review queue

with tab_review:
    t1 = st.session_state.get("t1")
    if not t1:
        st.info("Run the pipeline in **Digitize forms** first; flagged fields will appear here.")
    else:
        out = t1["output"]
        queue = []
        for p in out["pages"]:
            form = p.get("form")
            if not form:
                continue
            for k, label in TEXT_FIELDS:
                f = form[k]
                if f["needs_review"] and "empty" not in f["flags"]:
                    queue.append({"Page": p["page_number"], "Field": label, "key": k, "Machine reading": f["raw"] or "—",
                                  "Alternatives": ", ".join(f["alternatives"]), "Why flagged": ", ".join(f["flags"]),
                                  "Corrected value": f["value"] or f["raw"] or ""})
            for k, label in (("location_cwa", "Location (CWA)"), ("shift", "Shift")):
                if form[k]["needs_review"]:
                    queue.append({"Page": p["page_number"], "Field": label, "key": k, "Machine reading": form[k]["value"] or "—",
                                  "Alternatives": "", "Why flagged": ", ".join(form[k]["flags"]),
                                  "Corrected value": form[k]["value"] or ""})
            for it in form["checklist"]:
                if it["response"]["needs_review"]:
                    queue.append({"Page": p["page_number"], "Field": f"Checklist {it['item_no']}", "key": f"checklist:{it['item_no']}",
                                  "Machine reading": it["response"]["value"] or "—", "Alternatives": "S / R / I / N",
                                  "Why flagged": ", ".join(it["response"]["flags"]), "Corrected value": it["response"]["value"] or ""})
            for i, it in enumerate(form["expectation_questions"]):
                if it["response"]["needs_review"]:
                    queue.append({"Page": p["page_number"], "Field": it["question"], "key": f"expectation:{i}",
                                  "Machine reading": it["response"]["value"] or "—", "Alternatives": "yes / no / n/a",
                                  "Why flagged": ", ".join(it["response"]["flags"]), "Corrected value": it["response"]["value"] or ""})
        if not queue:
            st.success("Nothing to review - every extracted field passed validation (or was confirmed by two engines).")
        else:
            st.markdown(f"**{len(queue)} field(s) need a human decision.** Edit the *Corrected value* column, "
                        "then apply. Corrections are tagged `human_corrected` - in production they also feed "
                        "the training set.")
            edited = st.data_editor(pd.DataFrame(queue), hide_index=True, width="stretch",
                                    disabled=["Page", "Field", "Machine reading", "Alternatives", "Why flagged"],
                                    column_config={"key": None})
            if st.button("Apply corrections", type="primary"):
                fixed = json.loads(json.dumps(out))
                problems = []
                for row in edited.to_dict("records"):
                    page = next(p for p in fixed["pages"] if p["page_number"] == row["Page"])
                    form, key, val = page["form"], row["key"], str(row["Corrected value"]).strip() or None
                    if key.startswith("checklist:"):
                        if val and val.upper() not in "SRIN":
                            problems.append(f"Page {row['Page']} {row['Field']}: use S, R, I or N")
                            continue
                        tgt = form["checklist"][int(key.split(':')[1]) - 1]["response"]
                        val = val.upper() if val else None
                    elif key.startswith("expectation:"):
                        if val and val.lower() not in ("yes", "no", "n/a"):
                            problems.append(f"Page {row['Page']} {row['Field']}: use yes, no or n/a")
                            continue
                        tgt = form["expectation_questions"][int(key.split(':')[1])]["response"]
                        val = val.lower() if val else None
                    else:
                        tgt = form[key]
                        tgt["engine"] = "human"
                    tgt["value"], tgt["needs_review"] = val, val is None
                    tgt["flags"] = tgt.get("flags", []) + ["human_corrected"]
                for page in fixed["pages"]:
                    if page["form"] and page["status"] == "needs_review":
                        still = [r for r in edited.to_dict("records") if r["Page"] == page["page_number"]
                                 and not str(r["Corrected value"]).strip()]
                        if not still and not page["quality"]["warnings"]:
                            page["status"], page["review_fields"] = "ok", []
                if problems:
                    st.error("Not applied: " + "; ".join(problems))
                else:
                    st.session_state["corrections"] = fixed
                    st.success("Corrections applied.")
            if st.session_state.get("corrections"):
                st.download_button("Download corrected JSON", json.dumps(st.session_state["corrections"], indent=2),
                                   file_name="forms_output_reviewed.json", mime="application/json")


# ============================================================================ tab 3: robustness lab

DEGRADATIONS = {
    "Skew": None,  # uses the slider
    "Upside down (180°)": degrade.upside_down,
    "Mud stains": degrade.mud,
    "Blur": degrade.blur,
    "Low-resolution fax": degrade.low_res_fax,
    "Observer name crossed out": degrade.strike_observer,
    "Blank page": degrade.blank,
    "Black page": degrade.black,
    "Random noise": degrade.noise,
}

with tab_lab:
    st.markdown("Damage a sample page on purpose and compare the result with the clean extraction. "
                "**Pass rule:** every field must either match the clean read or be explicitly flagged / "
                "rejected - a changed value marked as trustworthy is a *silent error*.")
    forms = [p for p in sample_pages() if p.embedded_images] if SAMPLE_PDF.exists() else []
    if not forms:
        st.info(f"The robustness lab uses the sample forms - place `{SAMPLE_PDF.name}` in the project root.")
    c1, c2, c3 = st.columns(3)
    page_idx = c1.selectbox("Sample page", range(len(forms)), format_func=lambda i: f"Page {forms[i].page_number}")
    kind = c2.selectbox("Degradation", list(DEGRADATIONS))
    angle = c3.slider("Skew angle (°)", -15.0, 15.0, 5.0, 0.5, disabled=kind != "Skew")

    if st.button("Degrade & run", type="primary", disabled=not forms):
        warm_models()
        vlm = make_vlm(use_vlm)
        src = forms[page_idx]
        fn = degrade.skew(angle) if kind == "Skew" else DEGRADATIONS[kind]
        clean_key = (page_idx, use_vlm)
        cache = st.session_state.setdefault("clean_cache", {})
        with st.spinner("Reading the clean page…"):
            if clean_key not in cache:
                cache[clean_key] = process_page(src, vlm).model_dump(mode="json")
        with st.spinner(f"Reading the degraded page ({kind.lower()})…"):
            damaged = fn(src.image)
            got = process_page(RawPage(src.page_number, damaged), vlm).model_dump(mode="json")
        st.session_state["lab"] = {"clean": cache[clean_key], "got": got, "image": thumb(damaged), "kind": kind}

    lab = st.session_state.get("lab")
    if lab:
        clean, got = lab["clean"], lab["got"]
        left, right = st.columns([2, 3], gap="large")
        left.image(lab["image"], caption=f"Degraded input: {lab['kind']}")
        with right:
            st.markdown(f"Result: {badge(got['status'])} "
                        f"<span class='tb-muted'>{got.get('rejection_reason') or ''}</span>", unsafe_allow_html=True)
            if got["status"] in ("rejected", "error"):
                st.success("Refused safely - no data emitted for a page the pipeline can't trust.")
            else:
                c, g = degrade.flatten(clean["form"]), degrade.flatten(got["form"])
                rows = []
                for k, (cv, _) in c.items():
                    gv, review = g.get(k, (None, True))
                    outcome = "Match" if gv == cv else "Flagged for review" if (review or gv is None) else "SILENT ERROR"
                    rows.append({"Field": k, "Clean read": cv, "Degraded read": gv, "Outcome": outcome})
                df = pd.DataFrame(rows)
                n_silent = int((df["Outcome"] == "SILENT ERROR").sum())
                cards([("Fields", len(df)), ("Match", int((df["Outcome"] == "Match").sum())),
                       ("Flagged", int((df["Outcome"] == "Flagged for review").sum())), ("Silent errors", n_silent)])
                (st.error if n_silent else st.success)(
                    f"{n_silent} silent error(s)." if n_silent else "PASS - no wrong value slipped through as trusted.")
                st.dataframe(df.style.map(lambda v: "color:#B42318;font-weight:700" if v == "SILENT ERROR"
                                          else "color:#B45F00" if v == "Flagged for review" else "",
                                          subset=["Outcome"]), hide_index=True, width="stretch")
            if got["quality"]["warnings"]:
                st.caption("Quality warnings: " + ", ".join(got["quality"]["warnings"]))

    report = ROOT / "outputs" / "task1_stress_report.json"
    if report.exists():
        with st.expander("Full stress-suite report (python stress_test.py)"):
            data = json.loads(report.read_text())
            st.dataframe(pd.DataFrame([{
                "Case": r["case"], "Result": "PASS" if r["pass"] else "FAIL", "Status": r.get("status", ""),
                "Matched": f"{r['matched']}/{r['total']}" if r.get("total") else "",
                "Flagged": ", ".join(r.get("flagged") or []) if isinstance(r.get("flagged"), list) else "",
                "Detail": r.get("detail") or ""} for r in data]), hide_index=True, width="stretch")


# ============================================================================ tab 4: P6 classifier

TIER_ORDER = ["cache", "rule", "retrieval", "llm", "rule_hint", "unresolved"]
TIER_LABEL = {"cache": "Verified cache", "rule": "Keyword rules", "retrieval": "Hybrid retrieval (auto)",
              "llm": "Gemini RAG", "rule_hint": "Rule hint - needs review", "unresolved": "Needs review"}


@st.cache_resource(show_spinner="Loading the embedding model…")
def p6_retriever(run_id: str, bank: tuple):
    """Index of the confidently labelled activities from the last run (for 'try your own')."""
    keys, names, norms, prefixes, labels = (list(x) for x in zip(*bank)) if bank else ([], [], [], [], [])
    return HybridRetriever(keys, names, norms, prefixes, labels)


def neighbour_table(neigh: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Similar labelled activity": n["name"], "Client prefix": n["prefix"],
                          "Its label": n["label"], "Similarity": n["similarity"]} for n in neigh])


with tab_p6:
    st.markdown("Classify Primavera P6 activities into the fixed Discipline / Sub-Discipline rubric. "
                "Rows are deduplicated to canonical activities, then resolved by **keyword rules → hybrid "
                "retrieval over labelled examples → Gemini with retrieved evidence (RAG)**. The model can "
                "only answer with one of the 41 rubric labels or *UNCERTAIN*; every answer is re-validated.")
    c1, c2, c3 = st.columns([1, 2, 1.3])
    src = c1.radio("Schedule", ["Sample schedule (provided)", "Upload CSV"] if SAMPLE_CSV.exists() else ["Upload CSV"],
                   label_visibility="collapsed")
    up = c2.file_uploader("CSV with 'Activity ID' and 'Activity Name'", type=["csv"]) if src == "Upload CSV" else None
    if src != "Upload CSV":
        c2.caption(f"`{SAMPLE_CSV.name}` - 4,499 activities.")
    use_rag = c3.toggle("AI step: Gemini RAG", value=gemini_key, disabled=not gemini_key,
                        help="Activities not settled by rules or retrieval are sent to Gemini together with their "
                             "most similar labelled activities. Off = they stay in the review queue.")

    if st.button("Classify schedule", type="primary", disabled=(src == "Upload CSV" and up is None)):
        raw = pd.read_csv(io.BytesIO(up.getvalue()) if up else SAMPLE_CSV, encoding="utf-8-sig", dtype=str).fillna("")
        missing = {"Activity ID", "Activity Name"} - set(raw.columns)
        if missing:
            st.error(f"CSV is missing column(s): {', '.join(sorted(missing))}")
        else:
            bar = st.progress(0.0, text="Starting…")
            res_df, stats = classify(raw, llm_provider="gemini" if use_rag else "none",
                                     progress=lambda stage, frac: bar.progress(min(frac, 1.0), text=stage))
            bar.empty()
            st.session_state["p6"] = (res_df, stats, f"{len(res_df)}-{pd.Timestamp.now().value}")

    if st.session_state.get("p6"):
        res_df, stats, run_id = st.session_state["p6"]
        details = stats["details"]
        by_src = stats["rows_by_source"]
        n = stats["rows"]
        auto = sum(by_src.get(s, 0) for s in ("cache", "rule", "retrieval", "llm"))
        cards([("Activities", f"{n:,}"), ("Unique to classify", stats["unique_keys_after_normalisation"]),
               ("Classified", f"{auto / n:.1%}"), ("By Gemini RAG", f"{by_src.get('llm', 0):,} rows"),
               ("Needs review", f"{stats['rows_needing_review']:,} rows")])
        st.caption(f"Retrieval index: {stats['retrieval_index_size']} labelled activities · "
                   f"LLM: {stats['llm_models']} ({stats['llm_items_sent']} activities sent)")

        a, b = st.columns(2)
        with a:
            st.markdown("**Rows resolved by each tier** (in pipeline order)")
            funnel = pd.Series({TIER_LABEL[t]: by_src.get(t, 0) for t in TIER_ORDER if by_src.get(t, 0)}, name="rows")
            st.bar_chart(funnel, color=NAVY, horizontal=True, sort=False)
        with b:
            st.markdown("**Rows per discipline**")
            st.bar_chart(res_df["Discipline"].fillna("(review queue)").value_counts().rename("rows"),
                         color=ORANGE, horizontal=True)

        f1, f2, f3 = st.columns(3)
        disc = f1.multiselect("Discipline", sorted(res_df["Discipline"].dropna().unique()))
        srcs = f2.multiselect("Resolved by", [t for t in TIER_ORDER if t in set(res_df["source"])],
                              format_func=TIER_LABEL.get)
        only_review = f3.toggle("Only rows needing review")
        view = res_df
        if disc:
            view = view[view["Discipline"].isin(disc)]
        if srcs:
            view = view[view["source"].isin(srcs)]
        if only_review:
            view = view[view["needs_review"]]
        cols = ["Activity ID", "Activity Name", "Discipline", "Sub-Discipline", "confidence", "source",
                "matched_by", "needs_review", "evidence"]
        st.dataframe(view[cols], hide_index=True, width="stretch", height=420, column_config={
            "confidence": st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0, max_value=1),
            "source": st.column_config.TextColumn("Resolved by"),
            "matched_by": st.column_config.TextColumn("Why", width="large"),
            "evidence": st.column_config.TextColumn("Retrieved evidence (top 3)", width="large")})
        st.download_button("Download classified CSV", res_df[cols].to_csv(index=False), "p6_classified.csv", "text/csv")

        # ---------------------------------------------------------------- explain one activity
        st.markdown("#### Explain an activity")
        st.caption("Activities that went past the keyword rules: what retrieval found and what the model decided.")
        units = res_df.drop_duplicates("key").set_index("key")
        explain_keys = sorted(details, key=lambda k: (units.at[k, "source"] != "llm", k))
        if explain_keys:
            pick = st.selectbox("Activity", explain_keys, format_func=lambda k: (
                f"{units.at[k, 'Activity Name']}  [{units.at[k, 'prefix']}]  →  "
                f"{(units.at[k, 'Discipline'] or '—')} :: {(units.at[k, 'Sub-Discipline'] or 'review')}"))
            d = details[pick]
            left, right = st.columns([3, 2], gap="large")
            with left:
                st.markdown("**1 · Retrieved labelled examples** (semantic similarity)")
                st.dataframe(neighbour_table(d["neighbours"]), hide_index=True, width="stretch", column_config={
                    "Similarity": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
                st.markdown("**2 · Candidate labels from those examples:** " + ", ".join(f"`{c}`" for c in d["candidates"]))
            with right:
                st.markdown("**3 · Decision**")
                u = units.loc[pick]
                if u["source"] == "retrieval":
                    st.success(f"Auto-labelled by retrieval: **{u['Discipline']} :: {u['Sub-Discipline']}**  \n{u['matched_by']}")
                elif d.get("llm"):
                    dec = d["llm"]
                    shown = st.success if dec["label"] != "UNCERTAIN" and dec["confidence"] == "high" else st.warning
                    shown(f"**{dec['label']}** · confidence *{dec['confidence']}*  \n{dec['reason']}  \n`{dec['model']}`")
                else:
                    st.info("Not sent to an LLM in this run - stays in the review queue.")
                st.caption(f"Final: {u['Discipline'] or '—'} :: {u['Sub-Discipline'] or '—'} · "
                           f"{'needs review' if u['needs_review'] else 'accepted'} · {int((res_df['key'] == pick).sum())} rows")

        # ---------------------------------------------------------------- try your own
        st.markdown("#### Try your own activity")
        t1, t2, t3 = st.columns([3, 1, 1])
        q_name = t1.text_input("Activity name", placeholder="e.g. Install VESDA pipework in data hall 2")
        q_prefix = t2.selectbox("Client prefix", ["ARC", "CON", "ELE", "MEC", "SEC", "TELE", "GEN"])
        t3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if t3.button("Classify", disabled=not q_name.strip()):
            norm = normalize(q_name)
            st.markdown(f"Normalised: `{norm}`")
            r = rules.match(norm, q_prefix)
            if r and r.confidence >= rules.ACCEPT:
                st.success(f"**Keyword rule `{r.rid}`** → {r.label} (confidence {r.confidence:.2f}). Stops here - no AI needed.")
            else:
                if r:
                    st.info(f"Low-confidence rule hint `{r.rid}` → {r.label} ({r.confidence:.2f}) - passed on as a hint.")
                bank_rows = res_df[res_df["source"].isin(["rule", "cache"])].drop_duplicates("key")
                bank = tuple(zip(bank_rows["key"], bank_rows["Activity Name"], bank_rows["norm"], bank_rows["prefix"],
                                 bank_rows["Discipline"] + " :: " + bank_rows["Sub-Discipline"]))
                ret = p6_retriever(run_id, bank)
                neigh = ret.search_many([norm], [q_prefix])[0]
                st.dataframe(neighbour_table([x.__dict__ for x in neigh]), hide_index=True, width="stretch",
                             column_config={"Similarity": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
                lab = auto_accept(neigh, q_prefix)
                if lab:
                    st.success(f"**Retrieval auto-accept** → {lab} (top neighbours unanimous, similarity "
                               f"{neigh[0].similarity:.2f} ≥ {P6.AUTO_ACCEPT_SIM})")
                elif use_rag:
                    item = p6llm.Item("query", q_name, q_prefix, r.label if r else None,
                                      [{"name": x.name, "client_prefix": x.prefix, "label": x.label,
                                        "similarity": round(x.similarity, 2)} for x in neigh], candidate_labels(neigh))
                    with st.spinner("Asking Gemini with the retrieved evidence…"):
                        dec, used = p6llm.classify_items([item])
                    d = dec.get("query")
                    if d is None:
                        st.warning(f"No answer from the model ({used}) - this activity would go to the review queue.")
                    else:
                        (st.success if d.label != "UNCERTAIN" and d.confidence == "high" else st.warning)(
                            f"**{d.label}** · confidence *{d.confidence}*  \n{d.reason}  \n`{d.model}`")
                else:
                    st.info("Retrieval alone is not confident enough; with the AI step off this goes to review.")


# ============================================================================ tab 5: about

with tab_about:
    st.markdown(f"""
#### Task 1 - field data digitisation
Each kind of content is read by the cheapest tool that is reliable for it, and every stage can refuse:

| Stage | What happens | Refuses when |
|---|---|---|
| Ingest | each page rendered separately at {C.RENDER_DPI} dpi | file / page is corrupt → `error` |
| Quality gate | void stamp, ink ratio, blur score | void, blank → `rejected` |
| Structure | deskew, rule-grid detection (no OCR yet), orientation search | no table / no printed labels → `rejected` |
| Layout | every field = the cell next to its printed label | label missing → field flagged |
| Read | checkboxes by ink density · printed text by Tesseract · handwriting by TrOCR | low confidence / ambiguous → flagged |
| Validate | dates, IDs, names, form registry, strike-through, row alignment | invalid → `value: null`, raw kept |
| Escalate | only flagged handwriting crops → Gemini; accepted only if engines agree | disagreement → human review |

#### Task 2 - P6 classification (hybrid RAG)
`4,499 rows → 374 canonical activities → 87% resolved by keyword rules → retrieval over labelled
examples (auto-accept only when neighbours are unanimous and ≥ {P6.AUTO_ACCEPT_SIM} similar) →
Gemini decides the rest with the retrieved examples as evidence`.

| Measured on the sample (`evaluate.py`) | Result |
|---|---|
| Search the 41 label names (BM25 / semantic) | 9% / 21% top-1 - vocabulary mismatch |
| Search labelled examples, semantic | 76% top-1, 94% top-5 |
| Hybrid with BM25 | never better here - rules already use the exact keywords |
| Retrieval auto-accept at the chosen threshold | ~98% precision |

The rubric is enforced as an enum of the 41 valid *Discipline :: Sub-Discipline* pairs + `UNCERTAIN`.

Full write-up: `README.md`.
""")

st.markdown("<div class='tb-footer'>Prototype built by Rahul Verma for the Teknobuilt AI-ML Engineering "
            "assessment · not an official Teknobuilt product.</div>", unsafe_allow_html=True)
