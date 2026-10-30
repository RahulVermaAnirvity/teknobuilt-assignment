# Construction Data Automation: AI-ML Engineering Assessment

**Submission by Rahul Verma** for the Teknobuilt AI-ML Engineering assessment.

This repository contains working solutions to both assessment tasks, plus a branded web
console that runs them:

| | Task | What's delivered | Headline result |
|---|---|---|---|
| 1 | **Field Data Digitization Pipeline** (handwritten safety forms → JSON) | Python pipeline, output JSON, resilience strategy, stress-test harness, optional Gemini vision escalation, parallel page processing with Celery | Every value on all 3 sample forms extracted correctly; the "FAILED TEST FORM" is rejected; **14/14 stress cases with zero silent wrong values** |
| 2 | **P6 Schedule Batch Classification** (activities → Discipline / Sub-Discipline) | Architecture document (this README) **plus** a working hybrid-RAG classifier and an evaluation harness | 4,499 rows → 374 unique activities → **92.2% of rows auto-classified**, the rest routed to review; never outside the 41-label rubric |
| – | **Demo console** | Streamlit app (`app.py`) in Teknobuilt branding: run both tasks, inspect evidence, review queue, robustness lab | `docker compose up` → http://localhost:8501 |

---

## Contents

1. [Quick start with Docker (recommended)](#1-quick-start-with-docker-recommended)
2. [Local setup without Docker](#2-local-setup-without-docker)
3. [Running everything](#3-running-everything)
4. [Configuration (`.env`)](#4-configuration-env)
5. [Repository structure](#5-repository-structure)
6. [Task 1: Field Data Digitization Pipeline](#6-task-1-field-data-digitization-pipeline)
7. [Task 2: P6 Schedule Batch Classification (Architecture)](#7-task-2-p6-schedule-batch-classification-architecture)
8. [Troubleshooting](#8-troubleshooting)
9. [Security, data and honesty notes](#9-security-data-and-honesty-notes)

---

## 1. Quick start with Docker (recommended)

Docker runs everything in containers, so **nothing is installed on your machine and nothing
can conflict with your Python setup**. You only need
[Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose v2).

```bash
git clone <this-repo-url> teknobuilt-assignment
cd teknobuilt-assignment

# Optional: enable the AI steps (Gemini). Without a key everything still runs;
# unclear items simply go to the human-review queue.
cp .env.example .env            # Windows PowerShell: Copy-Item .env.example .env
#   then edit .env and set GEMINI_API_KEY=...   (free key: https://aistudio.google.com/apikey)

docker compose up --build
```

Open **http://localhost:8501**.

`docker compose up` starts three containers:

| Container | What it is |
|---|---|
| `teknobuilt-app` | the Streamlit console (port 8501) |
| `teknobuilt-worker` | Celery workers that process form pages in parallel (2 pages at a time by default) |
| `teknobuilt-redis` | Redis: the task queue and result store |

* **First build:** about 5–10 minutes. It downloads CPU PyTorch and bakes the two
  open-source models into the image, so later starts are fast and need no internet for
  the models.
* **More pages at once:** `PAGE_WORKERS=4 docker compose up` (each worker needs ~1 GB RAM).
* **Stop:** `Ctrl+C`, then `docker compose down`. Add `-v` to also delete the Redis data
  and uploads.
* **Run the command-line tools inside the image:**
  ```bash
  docker compose run --rm -w /app/task1_form_digitization app python run_pipeline.py ../Sample_input.pdf -o /app/data/out.json
  docker compose run --rm -w /app/task2_p6_classification app python classify_schedule.py ../P6_Schedule_Sample.csv -o /app/data
  ```

---

## 2. Local setup without Docker

Use this if you want to run the code directly. Everything installs into an **isolated
virtual environment (`.venv`)** inside the project folder, never into your global Python,
with the exact package versions this project was tested with.

### Prerequisites

| Requirement | Version | Install |
|---|---|---|
| Python | **3.11** (3.10–3.12 supported) | https://www.python.org/downloads/ |
| Tesseract OCR | 5.x | Windows: `winget install UB-Mannheim.TesseractOCR` · macOS: `brew install tesseract` · Ubuntu/Debian: `sudo apt-get install -y tesseract-ocr` |
| Docker *(optional)* | any recent | only for Redis if you want multi-process Celery workers |

Disk: about 2.5 GB for the environment and models. RAM: 4 GB minimum, 8 GB+ recommended.

### One-command setup

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
bash scripts/setup.sh
source .venv/bin/activate
```

The setup script:
1. finds a compatible Python and creates `.venv`;
2. installs the pinned `requirements.txt` (CPU-only PyTorch, about 200 MB rather than 2 GB);
3. creates `.env` from `.env.example` if it doesn't exist yet;
4. runs `scripts/check_setup.py`, which checks every package, the Tesseract binary, the
   optional API key (it reports only "set" / "not set", never the value) and Redis, then
   downloads the two models (~350 MB, once).

A successful check ends with `All good`.

### Manual setup (equivalent)

```bash
python3.11 -m venv .venv
source .venv/bin/activate              # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                   # optional keys
python scripts/check_setup.py
```

---

## 3. Running everything

All commands assume the virtual environment is active and you are in the repository root.

### 3.1 Web console (both tasks)

```bash
streamlit run app.py                    # -> http://localhost:8501
```

| Tab | What you can do |
|---|---|
| **Digitize forms** | Run Task 1 on the sample PDF or your own PDF / image. Each page shows its status (*Accepted / Needs review / Rejected / Error*), the scan, every field with confidence, the engine that read it (TrOCR / Gemini / ink measurement), flags and alternative readings. Download the JSON. |
| **Review queue** | The human-in-the-loop step. Flagged fields are listed with the machine reading and the reason; correct them (input is validated, e.g. checklist answers must be S/R/I/N) and download the corrected JSON. |
| **Robustness lab** | Damage a sample page on purpose (skew angle, mud, blur, fax quality, crossed-out name, blank, noise …) and see, field by field, whether each value matched, was flagged, or became a silent error. |
| **P6 classifier** | Run Task 2 on the sample schedule or your own CSV: tier funnel, discipline chart, filterable table with the evidence behind every label, **Explain an activity** (retrieved examples + Gemini's decision) and **Try your own activity**. |
| **How it works** | An architecture summary of both tasks. |

Sidebar switches:
* **Escalate unclear fields to Gemini:** needs `GEMINI_API_KEY`.
* **Parallel pages on Celery (N pages at a time):** shown when workers are running (§3.3).

### 3.2 Task 1: command line

```bash
cd task1_form_digitization
python run_pipeline.py ../Sample_input.pdf -o ../outputs/task1_output.json            # fully local, no API
python run_pipeline.py ../Sample_input.pdf -o ../outputs/task1_output.json --vlm      # + Gemini escalation
python run_pipeline.py my_scan.jpg -o out.json                                        # images work too
python stress_test.py  ../Sample_input.pdf -o ../outputs/task1_stress_report.json     # robustness suite (~5 min)
python stress_test.py  ../Sample_input.pdf -o ../outputs/task1_stress_report_vlm.json --vlm
```

### 3.3 Task 1: parallel pages with Celery (faster, identical output)

**Option A, Redis (one worker process per page slot):**
```bash
docker compose up -d redis                                   # just the broker
# add to .env:  CELERY_BROKER_URL=redis://localhost:6379/0
cd task1_form_digitization
python start_workers.py -n 4                                 # terminal 1: 4 worker processes
python run_pipeline.py ../Sample_input.pdf -o out.json --celery   # terminal 2
```

**Option B, zero setup (no Docker, no Redis):** leave `CELERY_BROKER_URL` unset. The
same two commands then use a local filesystem queue (`.celery/`), with one worker process
and N page slots.

The web console detects running workers automatically.

### 3.4 Task 2: command line

```bash
cd task2_p6_classification
python classify_schedule.py ../P6_Schedule_Sample.csv -o ../outputs              # Gemini RAG if GEMINI_API_KEY is set
python classify_schedule.py ../P6_Schedule_Sample.csv -o ../outputs --llm none   # no API calls at all
python evaluate.py ../P6_Schedule_Sample.csv                                      # measure retrieval + calibrate (free)
python evaluate.py ../P6_Schedule_Sample.csv --llm-sample 40                      # + live RAG accuracy test
```

The input CSV needs two columns: `Activity ID` and `Activity Name`.

### 3.5 Outputs

| File | Produced by | Contents |
|---|---|---|
| `outputs/task1_output.json` | `run_pipeline.py` | **Task 1 deliverable**: one record per page, every field with value, raw reading, confidence, engine, flags |
| `outputs/task1_stress_report.json`, `…_vlm.json` | `stress_test.py` | per-case pass/fail, matched/flagged fields |
| `outputs/task2_classified.csv` | `classify_schedule.py` | every activity with Discipline, Sub-Discipline, confidence, tier, reason, evidence, `needs_review` |
| `outputs/task2_stats.json` | `classify_schedule.py` | tier funnel counts |
| `outputs/task2_llm_batch_requests.jsonl` | `classify_schedule.py` | the exact LLM payloads (with retrieved evidence), for audit / Claude Batch API |
| `outputs/task2_eval.json` | `evaluate.py` | retrieval comparison, auto-accept calibration curve, live RAG accuracy |

---

## 4. Configuration (`.env`)

Copy `.env.example` to `.env`. **Every setting is optional.** `.env` is git-ignored and
excluded from the Docker image; never commit it.

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(unset)* | Enables Gemini for Task 1 field escalation and the Task 2 RAG step. Free key: https://aistudio.google.com/apikey |
| `CELERY_BROKER_URL` | *(unset = local filesystem broker)* | e.g. `redis://localhost:6379/0` for multi-process workers |
| `VLM_PROVIDER` | `auto` | Task 1 escalation provider: `auto` \| `gemini` \| `claude` |
| `GEMINI_MODELS` | `gemini-flash-lite-latest,gemini-flash-latest,gemini-3.8-flash` | Task 1 model fallback order |
| `P6_LLM_PROVIDER` | `auto` | Task 2 LLM provider: `auto` \| `gemini` \| `claude` \| `none` |
| `P6_GEMINI_MODELS` / `P6_GEMINI_ESCALATION_MODELS` | see `p6classify/config.py` | Task 2 worker / second-opinion models |
| `P6_W_SEMANTIC`, `P6_W_BM25` | `1.0`, `0.0` | Hybrid retrieval fusion weights (measured, §7.5) |
| `P6_AUTO_ACCEPT_SIM` | `0.80` | Retrieval auto-accept similarity threshold (calibrated, §7.5) |
| `ANTHROPIC_API_KEY` | *(unset)* | Claude alternative for both tasks (Task 2 via the Message Batches API) |
| `TESSERACT_CMD` | auto-detected | Path to the tesseract binary if it isn't on `PATH` |
| `HANDWRITING_MODEL` | `microsoft/trocr-small-handwritten` | Swap in e.g. `microsoft/trocr-base-handwritten` |
| `PAGE_TIME_LIMIT_S` | `900` | Hard per-page limit for Celery tasks |
| `PAGE_WORKERS` *(docker compose)* | `2` | Pages processed at a time in Docker |

---

## 5. Repository structure

```
.
├── README.md                     ← this document (setup + both tasks' write-ups)
├── app.py                        ← Streamlit console (both tasks)
├── app_assets/                   ← Teknobuilt logo files (taken from the assessment brief)
├── .streamlit/config.toml        ← brand theme (navy #043448, orange #F08404)
├── requirements.txt              ← exact tested versions (Python 3.11)
├── .env.example                  ← configuration template (copy to .env)
├── Dockerfile, .dockerignore     ← one image for the app and the workers
├── docker-compose.yml            ← redis + worker + app
├── scripts/
│   ├── setup.ps1 / setup.sh      ← one-command isolated setup (.venv)
│   └── check_setup.py            ← environment verification + model download
├── task1_form_digitization/
│   ├── run_pipeline.py           ← CLI: local or --celery, optional --vlm
│   ├── start_workers.py          ← launch N Celery page workers
│   ├── stress_test.py            ← synthetic-degradation robustness suite
│   └── form_pipeline/
│       ├── ingest.py             ← PDF/image → page images; per-page error isolation
│       ├── preprocess.py         ← quality metrics, deskew, rotation
│       ├── layout.py             ← ruled-grid detection, OCR anchors, cell lookup
│       ├── checkbox.py           ← ink-density checkbox reading
│       ├── ocr.py                ← Tesseract (printed) + TrOCR (handwriting, top-2 beams)
│       ├── validate.py           ← field validators: the anti-hallucination layer
│       ├── vlm.py                ← Gemini / Claude escalation (+ mock that never invents)
│       ├── extract.py            ← per-page orchestration
│       ├── tasks.py              ← Celery app: one page = one task
│       ├── schema.py             ← pydantic output contract
│       └── config.py             ← every threshold in one place
├── task2_p6_classification/
│   ├── classify_schedule.py      ← CLI
│   ├── evaluate.py               ← measurement + calibration harness
│   └── p6classify/
│       ├── taxonomy.py           ← the closed rubric (single source of truth)
│       ├── normalize.py          ← canonical activity keys (dedup)
│       ├── rules.py              ← tier 1: keyword rules
│       ├── retrieval.py          ← tier 2: hybrid retrieval over labelled examples
│       ├── llm.py                ← tier 3: RAG prompt, enum schema, Gemini / Claude / mock
│       ├── pipeline.py           ← the cascade + fan-out
│       └── config.py             ← weights, thresholds, models
├── outputs/                      ← generated results (see §3.5)
├── Sample_input.pdf, template (1).png, P6_Schedule_Sample.csv   ← assessment inputs
```

---

## 6. Task 1: Field Data Digitization Pipeline

### 6.1 The problem

Scanned, handwritten safety-observation forms must become structured JSON: date, observer,
company, location (CWA), shift, timesheet/employee numbers, the S/R/I/N checklist, the
CCZJV yes/no/N/A questions and comments. Real scans are skewed, muddy, folded and
crossed out, and some pages are not valid forms at all. The pipeline must not crash
and must not invent data.

### 6.2 Results on the sample PDF

| Page | Form | Status | Notes |
|---|---|---|---|
| 1 | Breaking Containment | `ok` | all fields extracted |
| 2 | O&I Confined Space | `ok` | all fields extracted |
| 3 | Critical Devices | `ok` with `--vlm` | TrOCR read timesheet `49012` at only 0.73 confidence (< 0.80 gate), so it was escalated; Gemini independently read `49012` → `engines_agree` → accepted. Offline, the field stays `needs_review` rather than being trusted. |
| 4 | "FAILED TEST FORM" | `rejected` | `void_marker_in_text_layer:failed_test`; no data emitted |

I checked every value against the scans by hand, and all are correct: 3/3 dates,
observers, companies, timesheet and employee numbers, comments, locations and shifts;
25/25 checklist answers; 12/12 expectation answers.

Excerpt of `outputs/task1_output.json`:

```jsonc
{
  "page_number": 1, "status": "ok", "review_fields": [],
  "quality": { "ink_ratio": 0.034, "blur_score": 1458.2, "deskew_angle_deg": 0.0, "rotation_applied_deg": 0, ... },
  "form": {
    "observation_date": { "value": "2024-10-02", "raw": "2024-10-02", "confidence": 0.915,
                          "engine": "trocr:microsoft/trocr-small-handwritten", "needs_review": false, "flags": [] },
    "observer":        { "value": "John Davis", ... },
    "location_cwa":    { "value": "8000", "scores": { "1000": 0.0, ..., "8000": 0.0263 }, ... },
    "shift":           { "value": "day", ... },
    "checklist_title": { "value": "Breaking Containment - Life Saving Action", ... },
    "checklist": [ { "item_no": 1, "question": { "value": "Has the location to be cut or opened been visibly marked?" },
                     "response": { "value": "I", "confidence": 1.0, "scores": { "S": 0, "R": 0, "I": 0.019, "N": 0 } } }, ... ],
    "expectation_questions": [ { "question": "Has positive feedback provided?", "response": { "value": "yes" } }, ... ],
    "comments": { "value": "Safety measures were strictly followed.", ... }
  }
}
```

Every value carries `confidence`, `engine`, `flags` and `needs_review`. When a reading
fails validation, `value` is `null` and the machine's reading is kept in `raw` for the
reviewer. **Downstream systems consume `value` only.**

### 6.3 Architecture

```
 PDF / image
     │
 ┌───▼────────────┐  per-page try/except: one corrupt page can't kill the batch
 │ 1 Ingest       │  render @200 dpi grayscale, keep PDF text layer, SHA-1 per page
 └───┬────────────┘
 ┌───▼────────────┐  void stamp in text layer? ── reject
 │ 2 Quality gate │  ink ratio (blank / black), Laplacian blur score, noise
 └───┬────────────┘
 ┌───▼────────────┐  Hough deskew → morphological grid detection (cheap, no OCR)
 │ 3 Structure    │  no ruled table ─────────────────────────────── reject
 │   & orientation│  OCR anchor labels at 0°/180°/90°/270°, keep the best
 └───┬────────────┘  too few labels ("not this form") ───────────── reject
 ┌───▼────────────┐  every field = "cell right of / below a printed label"
 │ 4 Layout       │  checklist rows walked until the ruled table ends (8, 9 … 15 rows)
 └───┬────────────┘
 ┌───▼───────────────────────────────────────────────────────────────────────┐
 │ 5 Read   checkboxes   → ink density, decided relative to siblings (no AI)  │
 │          printed text → Tesseract (titles, questions)                      │
 │          handwriting  → TrOCR, local CPU (names, dates, numbers, comments) │
 └───┬───────────────────────────────────────────────────────────────────────┘
 ┌───▼────────────┐  type validators, confidence gate, beam-ambiguity check,
 │ 6 Validate     │  strike-through detector, form-title registry, row alignment
 └───┬────────────┘
 ┌───▼────────────┐  only fields still flagged: small crop → Gemini / Claude
 │ 7 Escalate     │  accepted only if it agrees with the local reading
 └───┬────────────┘
     ▼
 JSON (pydantic-validated): ok | needs_review | rejected | error
```

### 6.4 Key design decisions

**1. Use the cheapest reliable tool for each kind of content.** Layout is geometry,
checkboxes are a measurement, printed text is easy OCR, handwriting is hard recognition,
and only the leftovers need judgement. Sending whole pages to a vision LLM was rejected:
it costs money per page, it invents plausible values when asked to "fill in the form",
and it gives no audit trail for why a box was considered ticked.

**2. Structure-relative layout, not template coordinates.** The filled forms don't match
`template.png`: the title differs per form type, they have 8–9 questions instead of 15,
and row heights vary. Fixed crop boxes would silently read the wrong regions. The pipeline
detects the ruled grid (long, thin morphological openings, which handwriting, X marks and
mud blobs don't survive) and finds printed labels with OCR. Every field is "the cell to
the right of the *Observer* label", and so on. A different checklist length just works,
and a missing label fails loudly.

**3. No language model for checkboxes.** Ink density inside each box (borders trimmed,
specks smaller than a pen stroke discarded) is decided *relative to its siblings*, which
cancels page-wide mud or toner. It is deterministic, explainable (per-box `scores` are in
the output) and free.

**4. A local handwriting model, with the VLM as the exception.** Tesseract reads the printed
text at ~95% confidence but fails on handwriting (`2024-70-02`, `John Davie`). TrOCR
(`microsoft/trocr-small-handwritten`, open source, CPU) read every handwritten sample field
correctly. Gemini is called only for fields that fail the local gate, with a crop of about
600×80 px.

### 6.5 Resilience strategy: handling "garbage" inputs

The guiding rule: **a wrong value that looks right is worse than a missing value.**
Every stage can refuse, and refusal is data, never an exception.

| Threat | Defence | Outcome |
|---|---|---|
| Corrupt / truncated / encrypted file | `pymupdf.open` and every page render wrapped individually | `status: error` for that page; the batch continues |
| Explicitly voided form ("FAILED TEST FORM", "VOID", …) | text-layer stamp check before any processing | `rejected: void_marker_in_text_layer` |
| Blank page | ink ratio < 0.4% | `rejected: blank_page` |
| Black page / random noise / photo of something else | no ruled grid after line extraction. This runs *before* OCR, so garbage is rejected in ~70–180 ms | `rejected: form_grid_not_found` |
| A ruled page that isn't this form (incl. a scanned void form without a text layer) | fewer than 4 printed anchor labels at any orientation | `rejected: form_labels_not_found` |
| Skewed / upside-down / sideways scans | Hough deskew (±20°) + 0/180/90/270 orientation search scored by anchor hits | processed normally; angle reported in `quality` |
| Mud, coffee, toner specks | adaptive binarisation; ink threshold below mud grey; specks below pen-stroke size ignored; sibling-relative decisions | marks still read; ambiguous groups flagged `multiple_marks` |
| Blurry / low-res scan | Laplacian blur score; on a degraded page *no handwritten field is auto-accepted* (`degraded_scan`) | values kept in `raw`, page `needs_review` |
| Plausible-but-wrong OCR (e.g. `09` read as `08`) | TrOCR returns its top-2 beams; if a different reading is ≥80% as likely → `ambiguous_reading` + `alternatives` | flagged, alternative shown to the reviewer |
| Crossed-out text | strike-through detector; the VLM prompt transcribes only the final entry | `strikethrough_detected`, flagged |
| Illegal values | dates must parse and fall in 2000…today, IDs 3–10 digits, names name-like, day/month-ambiguous dates not guessed | `value: null`, raw kept, flagged |
| Unknown form variant | OCR'd title snapped to a registry of known form types; unseen title → `unknown_form_type` | new layouts never silently enter the data |
| A table rule fades (blur, fold) and two rows merge | answer box must overlap ≥70% with its question row; the 4 fixed CCZJV questions are identified by text, not position | an answer is never attached to the wrong question |
| Two boxes ticked | a single-select group needs a clear winner (≥2.5× runner-up) | `value: null`, `multiple_marks` |
| Duplicate scans | SHA-1 of the rendered page | second copy `rejected: duplicate_of_page_N` |
| A bug on one odd page | top-level per-page `except` | `status: error` with the exception text; never a crash |
| VLM hallucination | one cropped field only; schema allows `legible:false, text:null`; answer re-enters the same validators; accepted only on agreement | a VLM can't bypass validation |

### 6.6 Stress test (proof of the resilience strategy)

`stress_test.py` degrades the sample pages synthetically and compares with the clean
extraction. A case **passes** if every field either matches the clean run or is
explicitly flagged / rejected. It **fails** if any value changes while marked trustworthy,
i.e. a silent hallucination.

**Result: 14/14 pass with zero silent wrong values**, both offline and with Gemini
(`outputs/task1_stress_report.json`, `…_vlm.json`).

| Case | Outcome (offline) | Fields identical to clean | With Gemini |
|---|---|---|---|
| Skew −7° | `ok` | 21/21 | 21/21 |
| Skew +4° | `needs_review` | 19/21: the date's top reading `2024-102.02` fails validation and the correct `2024-10-02` is offered as the runner-up alternative | 20/21 (date recovered) |
| Upside-down | recovered via orientation search | 22/22 | 22/22, `ok` |
| Mud stains | `needs_review` | 13/21 (stained checklist boxes flagged) | 13/21 |
| Mud + skew 3° | `needs_review` | 20/21 | 21/21 |
| Gaussian blur | `needs_review` (`blurry_scan`) | 9/22 | 11/22 |
| Low-res fax | `needs_review` (`blurry_scan`) | 10/21 | 10/21 |
| Observer crossed out | `needs_review` (`strikethrough_detected`) | 20/21 | 21/21, still flagged |
| Blank / black / random-noise page | `rejected` | n/a | n/a |
| "FAILED TEST FORM" as a pure scan | `rejected: form_labels_not_found` | n/a | n/a |
| Truncated PDF / random bytes | recoverable pages processed / `error` | n/a | n/a |

The harness found real silent failures in the first version, each fixed at the root rather
than by tuning thresholds to the test:
* a blurred `2024-09-30` read as `2024-08-30` at 0.85 confidence → runner-up-beam check and
  degraded-scan gate;
* a faded rule merged two rows and shifted answers onto the wrong question → row-alignment
  check and content-based question identification;
* noise pages spent minutes in OCR → cheap grid check moved ahead of OCR;
* dirt specks counted as ticks → connected-component stroke filter.

### 6.7 Vision-model escalation: exact configuration

`form_pipeline/vlm.py`. Only handwritten fields still `needs_review` after local reading
are escalated, so clean scans cost zero or a couple of calls per page.

* **Gemini (default):** `google-genai` → `models.generate_content(...)`,
  `response_mime_type="application/json"`, `response_json_schema=FIELD_SCHEMA`,
  `temperature=0`, 45 s timeout. Models are tried in order
  (`gemini-flash-lite-latest` → `gemini-flash-latest` → `gemini-3.8-flash`). An overloaded
  model is skipped; quota exhaustion (429) stops escalation and the field stays flagged.
  Measured: ~1,200 input / ~45 output tokens and 2–7 s per field. The key is passed
  explicitly, so a stray `GOOGLE_API_KEY` can't override it.
* **Claude (alternative):** `claude-opus-5-5` via `client.beta.messages.create`,
  `output_config={"effort": "low", "format": {"type": "json_schema", ...}}`, server-side
  refusal fallback enabled.
* **System prompt (summary):** transcribe verbatim; don't correct, guess or infer from
  context; ignore struck-through text; if illegible or empty return
  `legible:false, text:null` ("null is always preferable to a guess").
* **Output schema:** `{text: string|null, legible: bool, struck_through_present: bool,
  confidence: high|medium|low}`, `additionalProperties: false`.
* **Acceptance policy: the VLM is a second reader, not an oracle.**
  * If it agrees with TrOCR's reading (or TrOCR's runner-up), it's accepted (`engines_agree`).
  * If TrOCR's reading was invalid and the VLM is highly confident on a non-degraded page,
    it's accepted and tagged `vlm_only` for audit.
  * Anything else (disagreement, low confidence, crossed-out text) goes to human review,
    with both readings shown.

### 6.8 Throughput: one page per Celery task

**Rule: speed must never cost accuracy.** Nothing about *how* a page is read was changed:
no coarser OCR, smaller images or batching that could alter model outputs. Pages run
side by side instead.

* `form_pipeline/tasks.py` defines one task, `forms.process_page(path, page)`. The worker
  renders the page with the same code as a local run (pixel-identical) and calls the
  unchanged `process_page()`.
* `run_pipeline.py --celery` dispatches one task per page, then applies the same ordering,
  duplicate detection and summary. A failed or timed-out task becomes an `error` page.
* Reliability: `acks_late` + `reject_on_worker_lost` (pages are re-queued if a worker dies),
  `prefetch_multiplier=1`, and a per-page time limit. Redis persists the queue (append-only file).

| Mode (4-page sample, 8-core laptop) | Wall-clock | Output |
|---|---|---|
| Local, 1 page at a time | 95.8 s (incl. ~20 s model load) | baseline |
| **Celery + Redis, 4 worker processes** | **36.0 s** | **byte-identical**: every value, confidence and flag |
| Celery, zero-setup broker, 1 process × 4 slots | 32.5 s | byte-identical |

Each of the 4 workers processed exactly one page (from the worker logs).

### 6.9 Limitations and next steps

* **Calibration needs real data.** Thresholds were tuned on 3 clean forms plus synthetic
  damage. A few hundred labelled real scans would calibrate the gates per field (target
  < 0.5% silent errors) and measure the review rate.
* **TrOCR-small** was chosen for CPU speed. `trocr-base-handwritten`, or a fine-tune on
  reviewer corrections, would reduce the review volume, and GPU workers would add throughput.
* **Folded / curled pages:** deskew is affine. A dewarp step (outer table border → template
  rectangle) is the next addition.
* **Checklist questions** are OCR'd. In production they would come from the form registry,
  with OCR used only to verify.

---

## 7. Task 2: P6 Schedule Batch Classification (Architecture)

### 7.1 The problem

Primavera P6 schedules hold tens of thousands of activities that must be tagged with a
**Discipline** and **Sub-Discipline** from a fixed rubric (12 disciplines, 41 valid pairs).
Clients often assign them wrongly. Running 50k rows one by one through an LLM is slow,
expensive and prone to inventing categories.

### 7.2 What the sample data says (drives the design)

* **4,499 rows but only 741 distinct names.** After stripping location and sequencing
  tokens (`COLO3`, `Cell 2`, `GL 15-22`, `(1/2)`, `Zone F`, `HEL16-…`) there are only
  **374 distinct classification problems**. Classifying rows is the wrong unit of work;
  classify *unique canonical activities* and fan the results out.
* **Some names only make sense in context.** `Testing`, `Cabling`, `Terminations` and
  `Dressing` appear under TELE, ELE, SEC and CON. The client's ID prefix disambiguates
  generic names.
* **The client's discipline is often wrong.** `ARC` rows contain plumbing (`Install DHW and
  DCW drainage pipework`), ductwork and telecom racks, so the prefix is a weak hint, never
  the answer.

### 7.3 Pipeline architecture (deliverable 1)

```
 P6 export (CSV, 50k+ rows)
        │
 ┌──────▼────────┐  canonicalise names (typos, location/phase tokens) → key = prefix|canonical
 │ Normalise &   │  4,499 rows → 374 keys on the sample (−91.7%)
 │ dedupe        │
 └──────┬────────┘
 ┌──────▼────────┐  key verified before → reuse (grows with every project)
 │ T0 Cache      │
 └──────┬────────┘
 ┌──────▼────────┐  ordered high-precision keyword rules, each bound to a valid label
 │ T1 Rules      │  confidence ≥ 0.85 → accept; lower → "hint" passed to the LLM
 └──────┬────────┘
 ┌──────▼────────┐  hybrid retrieval (semantic + BM25, RRF) over LABELLED EXAMPLES;
 │ T2 Retrieval  │  auto-accept only if top-3 neighbours unanimous + same prefix +
 │               │  cosine ≥ 0.80 (calibrated ≈98% precision); else → evidence for T3
 └──────┬────────┘
 ┌──────▼────────┐  residual keys + retrieved evidence (5 most similar labelled
 │ T3 LLM (RAG)  │  activities, candidate labels, rule hint); enum-constrained output;
 │               │  only "high" confidence auto-accepted
 └──────┬────────┘
 ┌──────▼────────┐  UNCERTAIN / non-high → stronger model; agreement → accept
 │ T4 Escalation │
 └──────┬────────┘
 ┌──────▼────────┐  label ∈ rubric? every key answered? else retry → human review
 │ Validate      │
 └──────┬────────┘
 ┌──────▼────────┐  join back to all rows: label, confidence, tier, reason, evidence,
 │ Fan-out       │  needs_review; verified answers → cache
 └───────────────┘
```

**Scaling.** Normalisation, cache and rules are vectorised and take seconds on 50k rows.
Retrieval is an in-memory index; a few thousand examples search in milliseconds, so no
vector database is needed at this scale. Only the residual keys reach an LLM, packed 30
per request. For large asynchronous runs the same payloads go to the **Claude Message
Batches API** (up to 100k requests, 50% price, results keyed by `custom_id`, so a partial
failure re-submits only the missing ids). For interactive use, Gemini is called
synchronously. Orchestration fits in one queue worker (the same Celery setup as Task 1).

### 7.4 Constraint enforcement: guaranteeing the rubric (deliverable 2)

Defence in depth: together these layers make an invalid label impossible to persist.

1. **Labels are pairs, not two free fields.** The rubric is flattened into 41 strings such
   as `"Electrical :: Cable Trays Drops"`, so a wrong combination like
   `Plumbing :: Cable Pulling` can't be expressed.
2. **Constrained decoding.** The response schema declares `label` as an **`enum` of the 41
   pairs + `"UNCERTAIN"`** (Gemini `response_json_schema`, Claude `output_config.format`).
   A new category isn't a reachable output.
3. **A sanctioned "I don't know".** `UNCERTAIN` routes vague names (`AZNG`, `RNG`,
   `Fit-Out`) to review instead of a confident-looking guess. On the sample, the model
   used it for exactly those.
4. **Code-side validation.** Every answer is re-checked against `taxonomy.LABEL_SET`, and
   every key must come back once. Otherwise it's retried, then sent to review. Hand-written
   rules go through the same `label()` constructor, which raises on a typo.
5. **One source of truth.** `taxonomy.py` generates the enum, the prompt text and the
   validator, so they can't drift apart.

### 7.5 Cost and accuracy optimisation (deliverable 3), with measurements

**Fewer questions beats cheaper questions.**

| Lever | Effect |
|---|---|
| Dedupe on canonical keys | 4,499 → 374 (−91.7%) |
| Persistent label cache | repeat activities across projects cost nothing |
| Keyword rules first | 328 of 374 keys = **87% of rows**, free and auditable (`matched_by` = rule id) |
| Retrieval auto-accept | only under calibrated, strict conditions |
| RAG instead of zero-shot | the LLM sees solved precedents, which measurably raises accuracy (below) |
| Batching + prompt caching | instructions and rubric paid once per 30 items; the identical prefix is cached |
| Batch API / cheap worker model + escalation | 50% discount; a strong model only for the unsure tail |

**Why retrieval searches *labelled examples*, and why BM25's weight is 0** (`evaluate.py`;
328 rule-labelled activities, same-name twins hidden from the index):

| Search against | Method | Top-1 | Top-5 recall |
|---|---|---|---|
| the 41 label names | BM25 | 9% | 16% |
| the 41 label names | semantic | 21% | 49% |
| labelled examples | BM25 only | 65.9% | 89.6% |
| labelled examples | hybrid 1 : 1 | 73.5% | 91.8% |
| labelled examples | hybrid 1 : 0.3 | 75.0% | 93.0% |
| labelled examples | **semantic only** | **76.2%** | **93.9%** |

Activity names and label names share almost no vocabulary, so the index is built from
*previously labelled activities*. BM25 never improved results here, because the exact
keywords it is good at are already consumed by the rules tier. The hybrid machinery stays
in place (`P6_W_BM25`) and should be re-evaluated on a human-labelled gold set. A
construction abbreviation dictionary (VESDA, NPW, LTHW, UGS, HAC, …) is appended to names
before embedding.

**Auto-accept calibration** (top-3 unanimous + same prefix):

| Cosine threshold | Activities auto-accepted | Precision |
|---|---|---|
| 0.70 | 89 | 97.8% |
| **0.80 (chosen)** | 53 | **98.1%** |
| 0.90 | 25 | 96.0% |

**Live RAG accuracy** (Gemini, 40 held-out activities whose labels and same-name twins
were removed from the index):

| | Result |
|---|---|
| Retrieval alone (top candidate) | 72.5% correct |
| **Gemini RAG** (same items, with retrieved examples) | **82.5% correct**, +10 points |
| Gemini answers marked `high` (25 of 40), the only ones auto-accepted | **96% correct** |

**Full run on the sample schedule** (`classify_schedule.py`, Gemini free tier):

| Tier | Unique activities | Rows |
|---|---|---|
| Keyword rules | 328 | 3,917 |
| Retrieval auto-accept | 0 (none of the 46 leftovers met the strict bar) | 0 |
| Gemini RAG: accepted (high confidence) | 10 | included below |
| Gemini RAG: answered, flagged for review | 28 | |
| Gemini RAG total | 38 | 526 |
| `UNCERTAIN` → review (`Testing` under CON, `Fit-Out`, `AZNG`, `RNG`, `EPMS`, `SLV/SCCR`, UMS fit-outs) | 8 | 56 |
| **Auto-classified** | | **4,146 rows (92.2%)** |
| **Needs review** | | **353 rows (7.8%)**, each with a suggested label, reason and evidence |

**Rough cost at 50k rows** (illustrative: ~5k unique keys, ~15% reaching the LLM, cached
system prompt): the naive approach (50k sequential calls to a large model) is ≈ $350 and
many hours; **this design is ~25 batched requests, cents, and minutes**. On the Gemini free
tier the sample costs nothing.

**Protecting accuracy over time:**
* **Provenance on every row:** tier, rule id / LLM reason, retrieved evidence, confidence.
* **A gold set + regression gate:** a few hundred human-labelled keys to measure each tier,
  re-tune thresholds and gate prompt or model changes (`evaluate.py` is the harness).
* **Active learning:** reviewer corrections enter the cache *and* the retrieval index, so
  the same mistake isn't repeated. Frequent LLM answers are promoted to rules, and a rule
  that keeps getting overridden is demoted to a hint.

### 7.6 Examples

| Activity (client prefix) | Result | Tier / why |
|---|---|---|
| `Install DHW and DCW drainage pipework` (ARC) | Plumbing :: Rough-In | rule; content beats the wrong prefix |
| `Installation of Primary Containment - Drops` (ELE) | Electrical :: Cable Trays Drops | rule ("drops" ordered before "primary") |
| `Splice` / `Dressing` (TELE) | Telecommunications :: Production Fiber | rule; generic names disambiguated by prefix |
| `Firestopping Service Openings` (ARC) | Architectural :: Interior Finishes | Gemini RAG, high confidence, accepted |
| `Install Secondary Containment for Alarms` (SEC) | Electrical :: Cable Trays Drops | Gemini RAG, high, accepted (the client's SEC is wrong) |
| `PV panels` (CON) | Structural :: Envelope (*review*) | Gemini RAG, medium, so a human decides |
| `AZNG`, `RNG`, `EPMS` (TELE) | `UNCERTAIN` → review | unexplained acronyms are not guessed |

**An honest example of why `high` answers are logged for audit:** Gemini accepted
`Level 1 Admin … Slab Prep & Pour` as *Civil :: Slab on Grade* with high confidence. An
upper-level slab is arguably *Structural*, which the rule hint suggested and the prompt
states. That's a judgement call a gold set would settle, and every such decision carries
its reason in `matched_by`.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `tesseract binary not found` | Install Tesseract (§2), or set `TESSERACT_CMD=/full/path/to/tesseract` in `.env` |
| `pip` tries to build or download huge CUDA packages | Make sure you install from `requirements.txt` (it points pip at the CPU-only PyTorch index) inside the fresh `.venv` |
| TensorFlow / Keras errors on import | You're outside the project venv with TensorFlow installed globally. Activate `.venv`; the project also sets `USE_TF=0` |
| Gemini `503 high demand` / timeouts | Free-tier capacity. The code retries, falls back to the next model, and otherwise leaves items for review. Re-run later |
| Gemini `429 RESOURCE_EXHAUSTED` | Daily free quota used up. The AI steps stop cleanly; everything else still runs |
| Console shows "0 pages at a time" for Celery | Workers aren't running (`python task1_form_digitization/start_workers.py -n 4`), or `CELERY_BROKER_URL` points to a Redis that isn't up (`docker compose up -d redis`) |
| Port 8501 or 6379 already in use | Stop the other process, or change the left side of the port mapping in `docker-compose.yml` |
| Docker build is slow the first time | Expected: it downloads PyTorch and the models once (~5–10 min). Later builds use the cache |
| First local run pauses for ~20–30 s | The handwriting model loads once per process (`check_setup.py` pre-downloads it) |

---

## 9. Security, data and honesty notes

* **Secrets:** API keys live only in `.env`, which is git-ignored and excluded from the
  Docker image. They're passed explicitly to the SDKs and never printed.
* **Privacy:** free-tier AI APIs may use inputs to improve their models. That's fine for this
  sample; real worker names and IDs belong on a paid tier with a no-training agreement, or
  on a local model (the whole pipeline runs offline without any key).
* **Measurement honesty:** Task 1 accuracy was checked by hand against the scans. For Task 2
  there is no official answer key: accuracy numbers are measured against the activities
  labelled by the high-precision rules (pseudo-gold). They're valid for *comparing* methods
  and calibrating thresholds, not as a claim of absolute accuracy. The rules themselves were
  written and audited against the same CSV.
* **Branding:** the console uses Teknobuilt's logo and colours taken from the assessment
  brief, and is labelled as an assessment prototype, not an official Teknobuilt product.
