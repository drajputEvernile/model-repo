# Technical architecture

The shape of the system, the data model, and **what every file in the repository
is for**.

Companion documents: [FLOW.md](FLOW.md) (what runs when),
[LOGIC.md](LOGIC.md) (how each decision is made), [API.md](API.md) (how to run
and call it).

---

## Contents

1. [System shape](#1-system-shape)
2. [Data model](#2-data-model)
3. [The shared workspace](#3-the-shared-workspace)
4. [File inventory](#4-file-inventory) ← role of every file
5. [Design decisions](#5-design-decisions)
6. [Known limits](#6-known-limits)

---

## 1. System shape

Two independently deployable services over one database and one shared volume.

```mermaid
flowchart TB
  subgraph EXT["External"]
    BLOB[("Azure Blob")]
    ADI[("Azure Document<br/>Intelligence")]
  end

  subgraph CP["core-pipeline · port 8001 · own compose"]
    direction TB
    API["api/main.py<br/>FastAPI"]
    CLI["cli.py"]
    ORCH["orchestrator/runner.py"]
    ST["stages/ × 8"]
    LIB["stages/lib/<br/>imaging · junk · member · dos"]
    DB["db/<br/>persistence + status"]
    API --> ORCH
    CLI --> ORCH
    ORCH --> ST --> LIB
    ST --> DB
  end

  subgraph RU["review-ui · Docker ports 4000/4001 · own compose"]
    direction TB
    RAPI["backend/app<br/>FastAPI viewer"]
    ADPT["adapters/<br/>local | postgres"]
    FE["frontend/src<br/>Vite + React"]
    FE --> RAPI --> ADPT
  end

  PG[("PostgreSQL — schema v8")]
  VOL[/"data/folders<br/>pages · ocr · imaging"/]

  BLOB --> ST
  ADI --> ST
  DB <--> PG
  ST --> VOL
  ADPT --> PG
  VOL -. "read-only" .-> ADPT

  style CP fill:#e8f0fe,stroke:#4a76c7
  style RU fill:#eaf6ec,stroke:#4c9a5b
  style EXT fill:#faf3e0,stroke:#c7a54a
```

**Why no HTTP between them.** The review UI is a viewer over finished work. If
it called the pipeline it would need the pipeline up to render a chart the
pipeline finished last week. Sharing the database and the volume means either
service can be redeployed, scaled or taken down alone.

| | core-pipeline | review-ui |
|---|---|---|
| Deploys | `core-pipeline/docker-compose.yml` | `review-ui/docker-compose.yml` |
| Ports | 8001 | 4000 (API), 4001 (web) in Docker; 8002 / 5174 locally |
| `data/folders` | read-write | **read-only** |
| Database | required | Production Mode only |
| Scaling | one process per chart chain | stateless, scales freely |

---

## 2. Data model

Schema v8 — `schema/v1.sql` (implemented) and `schema/v2.sql` (next phase).

```mermaid
erDiagram
  chart_list      ||--o{ page_list : "has"
  chart_list      ||--o{ pipeline_jobs : "logs"
  chart_list      ||--o| member_verification_summary : "decides"
  chart_list      ||--o{ manifest_member_list : "linked (nullable)"
  page_list       ||--o{ page_stage_status : "progress"
  page_list       ||--o{ ocr_results : "text × engine"
  page_list       ||--o| ocr_quality_results : "rotation + HW"
  page_list       ||--o{ blank_junk_classification : "verdict × pass"
  page_list       ||--o| member_extraction_results : "per-page verdict"
  page_list       ||--o| dos_extraction_results : "DOS (dates[] inline)"
  pipeline_stage  ||--o{ page_stage_status : "defines"
  manifest_member_list ||--o{ member_extraction_results : "matched"
```

### Core tables

| Table | Grain | Role |
|---|---|---|
| `pipeline_stage` | stage × pass | **The pipeline's shape as data.** Order, labels, whether a stage is orchestrated. Adding a stage is an INSERT. |
| `chart_list` | chart | Identity + lifecycle `status` + `current_stage`/`current_pass` + `output_path` (Processed/… write destination). `UNIQUE (chart_name)`. |
| `page_list` | page | One row per image. `image_sha256` for download idempotency / image-level dedup. `use_corrected` + `image_path` record which workspace file stages should read (`pages/…` vs `corrected-pages/…`). |
| `page_stage_status` | page × stage × pass | Progress. Replaces v6's 11 status columns. Drives resume and status derivation. |
| `manifest_member_list` | record × member | The client roster, keyed on `record_id` — which **is** `chart_list.chart_name`. No `chart_id` column: the relationship is a join, so a sweep can precede ingest with nothing to link afterwards. |
| `page_ground_truth` | chart × page file | Client imaging labels. `chart_name` is the folder name; `page_number` is the file stem (`1` matches `1.jpg` / `1.png` / `1.tif`). No `chart_id`, so the spreadsheet can load before the chart exists. |

### Result tables

| Table | Key | Written by |
|---|---|---|
| `ocr_results` | `(page_id, ocr_type)` | stages 2, 4, 5 |
| `ocr_quality_results` | `page_id` | stage 1. Rotation + HW + measured quality (`quality_tag` / `quality_score` / `quality_detail`). Not redundant with the CSVs — CSVs are rebuilt from this table; Production Mode reads it. |
| `blank_junk_classification` | `(page_id, pass_no)` | stages 3, 6 |
| `member_extraction_results` | `page_id` | stage 7 |
| `member_verification_summary` | `chart_id` | stage 7 |
| `dos_extraction_results` | `page_id` | stage 8. Page/document pairs are single-valued columns; every date found sits in the `dates` JSONB array. |
| `pipeline_jobs` | run | every stage |

### Views

| View | Purpose |
|---|---|
| `v_chart_stage_progress` | Per-chart per-stage page rollup. Chart status reads this. |
| `v_page_blank_junk_final` | The one winning blank/junk row per page. |
| `consolidated_chart_results` | One wide row per chart for reporting. |
| `pipeline_stage_performance` | Stage timings and success rates. |

### Constraints that carry meaning

| Constraint | Prevents |
|---|---|
| `chart_list UNIQUE (chart_name)` | Two concurrent ingests creating two charts with one name. |
| `blank_junk UNIQUE (page_id) WHERE is_final` | Two "final" verdicts for one page. |
| `junk_subtype CHECK` | A label the UI cannot render reaching the database. |
| `page_stage_status UNIQUE (page_id, stage_name, pass_no)` | Duplicate progress rows; makes upsert-on-conflict safe. |
| `manifest` two partial unique indexes | Duplicate roster rows, with and without a MemberID. |

### Tables present but not yet written

`page_classification`, `chunk_results`, `encounter_type_results`,
`page_sequencing_results`, `rejection_results`, `provider_signature_results`,
`invoice_matching_results`, `ground_truth_csv`, `model_registry`,
`field_accuracy_log`, `model_accuracy_snapshots`, `manual_review`, `audit_log`,
`users`.

`manual_review` / `rejection_results` / `audit_log` / `users` are the review-write
phase, deliberately not built — the UI is a viewer. The rest are registered in
`pipeline_stage` with `is_phase1 = FALSE`, so they sit outside the status rollup
until their stage modules land.

---

## 3. The shared workspace

```
review-ui/data/folders/<chart_name>/
├── pages/
│   └── 1.jpg … N.jpg              written by intake; served by review-ui
├── ocr/
│   ├── <chart>_prelim.txt         ===== <page> ===== delimited blocks
│   ├── <chart>_final1.txt         same format
│   └── <chart>_final2.json        {recordId, pageCount, pages[{fileName, content}]}
└── imaging/
    ├── <chart>_rotation.csv
    ├── <chart>_hw_printed.csv
    ├── <chart>_junk.csv
    ├── <chart>_member_extraction.csv
    ├── <chart>_member_verification.csv
    ├── <chart>_member_v1_compare.csv    reference column order, for diffing
    └── <chart>_dos.csv
```

Everything here is **rebuilt from the database** each run, so a resumed run never
leaves a half-written file.

The `===== <page_name> =====` marker is a real contract: the DOS driver's
`UI_PAGE_MARKER_RE` splits on exactly it. Pinned by
`tests/test_contracts.py::test_dos_splitter_reads_the_same_marker`.

**Precedence in review-ui Local Mode:** per-chart `imaging/*.csv` first, then the
legacy combined packs (`01-ocr-extraction/output/…`) from the older pipeline.
Per-chart always wins; the fallback exists so pre-existing `05-imaging-ui` data
still renders.

---

## 4. File inventory

### Repository root

| File | Role |
|---|---|
| `README.md` | Project overview and quick start. |
| `README.md` | Short orientation; detail lives in `docs/`. |
| `.gitignore` | Excludes venvs, caches, `node_modules`, `dist`, and macOS `._*` / `.DS_Store` droppings. |
| `docs/` | This documentation set. |

### `schema/`

| File | Role |
|---|---|
| `v1.sql` | **What is implemented.** Required. Includes `page_classification`, `encounter_type_results`, `page_sequencing_results`, and 12 phase-1 stages. |
| `v2.sql` | **Next phase. Nothing implemented.** Optional; apply after v1.sql. Proposals only (chunking, rejection, models, …) + `rejection_logic` stage. |
| `patch_output_path.sql` | **Existing DBs only.** Idempotent upgrade: `output_path`, classification/encounter/sequencing tables, stage registry, page_stage_status seeds. |

### `core-pipeline/` — top level

| File | Role |
|---|---|
| `capabilities.py` | What each optional feature can actually do right now — blob, Azure DI, the key/value extraction — and the one precondition each is missing. Read by both the startup banner and `GET /health`, so they cannot disagree. Configuration only; opens no sockets, except `probe_blob()` which startup calls once, bounded. |
| `stages/lib/image_preprocess/osd.py` | Coarse page orientation from Tesseract OSD — the clockwise rotation to apply, with a confidence floor, declining rather than guessing on a sparse page. Replaces the geometric detector's coarse step, which recovered 0 of 6 sideways pages at confidence 1.000. |
| `stages/lib/image_preprocess/stage.py` | **Stage 1**, moved ahead of OCR so every pass reads an upright page. Always measures orientation/tilt/mirror; writes `corrected-pages/<n>.jpg` only when `ROTATION_CORRECTION_ENABLED` and only for pages that change. `rotation_applied` means a corrected file exists, not that the page looked crooked. |
| `config.py` | Every environment-driven setting in one place: database URL, data roots, Azure credentials, the extraction weights root (`EXTRACTION_MODELS_ROOT`; the model version is pinned to v002), `STAGE_WORKERS`, and the `chart_dir` / `pages_dir` / `ocr_dir` / `imaging_dir` path helpers. |
| `cli.py` | Command-line entry: `serve`, `run`, `batch`, `write`, `rerun`, `stages`, `status`, `manifest`. Mirrors the API one-for-one, without the HTTP hop. |
| `requirements.txt` | Python dependencies for the service. |
| `requirements-extraction.txt` | The key/value extraction runtime (`gliner`, `torch`, `transformers`, `lightgbm`, `pandas`), pinned to the versions the extraction was built and trained on. Required; the Docker image always installs it. |
| `Dockerfile` | Runtime image. Installs Tesseract and the OpenCV/ONNX system libraries the reference modules need. |
| `docker-compose.yml` | Standalone deployment: ports, env, and the four volume mounts. |
| `.env.example` | Documented template for `.env`, with the consequence of leaving each optional service unset. |
| `__init__.py` | Package marker. |

### `core-pipeline/api/`

| File | Role |
|---|---|
| `main.py` | FastAPI app. Request models, the async 202 pattern, and the endpoints: `/health`, `/ready`, `/api/stages`, chart `run` / `batch` / `write` / status / `rerun`, manifest sweep and lookup, `/api/jobs`. Closes the connection pool on shutdown. |
| `__init__.py` | Package marker. |

### `core-pipeline/orchestrator/`

| File | Role |
|---|---|
| `runner.py` | `STAGE_CHAIN` — the nine stages in order — plus `run_pipeline_for_chart` (resume, `force`, `only`, `through`), `resolve_stage` (the one place a stage name is parsed) and `ingest_and_run`, which both `/run` and `/batch-run` go through. Refreshes chart status after each stage; aborts the chain on a stage exception, because every later stage reads what the failed one produced. |
| `__init__.py` | Package marker. |

### `core-pipeline/db/`

| File | Role |
|---|---|
| `__init__.py` | **The persistence layer.** Connection pool, `connect()`, hashing helpers, and every read/write: chart and page upserts, the `page_stage_status` helpers (`init_page_stages`, `set_page_stage`, `pages_needing_stage` ← resume), job rows, and one upsert per result table. All `ON CONFLICT`, never SELECT-then-UPDATE. |
| `chart_status.py` | Derives `chart_list.status` / `current_stage` / `current_pass` from `v_chart_stage_progress`. `compute_progress()` is pure, so it is directly testable. |
| `paths.py` | The disk contract: local page listing, the `===== page =====` combined-text writer/parser, the final2 JSON writer/parser, CSV write/append, and `imaging_csv()` naming. |
| `blob_store.py` | Azure Blob access: auth (Entra / key / connection string), `list_image_blobs`, `download_blob_to_path`, `chart_name_from_blob_path`. |

### `core-pipeline/jobs/`

| File | Role |
|---|---|
| `manifest_sweeper.py` | Batch manifest loader. Parses CSV/XLSX from a file, directory or blob prefix; recognises the column aliases; splits name parts; derives `run_id`/`batch_id` from the `R#_B#` filename; upserts on `record_id`. Creates no placeholder charts. |
| `__init__.py` | Package marker. |

### Stage runners

Shared plumbing stays in `core-pipeline/stages/` (`_support.py`) and `stages/utilities/`; each stage runner lives next to its engine under `stages/lib/<module>/`.

| File | Role |
|---|---|
| `_support.py` | Shared stage plumbing: the `stage_run()` context manager (job row, page load, resume set, job close), `mark_processing` / `mark_completed` / `mark_failed` / `mark_skipped`, and the shared eligibility rule. Keeps each stage about its actual work. |
| `utilities/download_blob.py` | **Intake.** Upserts the chart, downloads page images (skipping bytes already on disk), records SHA-256 + size, seeds `page_stage_status`, links manifest rows swept earlier. `import_local_folder()` is the local-source equivalent; `register_local_pages()` registers a folder already under `data/folders`. |
| `lib/ocr/stage_prelim.py` | **Stage 2.** Tesseract over every page (the corrected image when one exists), threaded to `STAGE_WORKERS`. Writes `ocr_results` and rebuilds `_prelim.txt`. |
| `lib/image_preprocess/stage.py` | **Stage 1.** Rotation, handwriting (ConvNeXt or RF), and measured quality analyzer. Writes `ocr_quality_results` plus rotation / hw / quality CSVs. |
| `lib/blank_junk/stage.py` | **Stages 3 and 7.** Both passes: eligibility, ±2-neighbor similarity duplicates, the subtype mapping into the schema's constrained vocabulary, `mark_blank_junk_final`, and a full CSV rewrite from the database. |
| `lib/ocr/stage_final1.py` | **Stage 4.** Docling+RapidOCR when ready; else RapidOCR-onnx only. Stores as `ocr_type='docling'` — the UI's "Final (OSS)" slot. Writes `section_header_candidates`. |
| `lib/ocr/stage_final2.py` | **Stage 5.** Azure Document Intelligence `prebuilt-read`, one shared client. Skips high-quality printed pages. The billed stage, so the resume path matters most here. |
| `lib/extraction/stage.py` | **Stage 6 (`kv_extract`).** Runs every extractor over the chart's Final2 word boxes, stages the result, and writes the headings into `section_headers`. See [EXTRACTION.md](EXTRACTION.md). |
| `utilities/gate_delta.py` | Adaptive skip_ocr: compare quality/rotation gate signatures and reopen only affected `page_stage_status` rows. |
| `lib/member/stage.py` | **Stage 8.** Plumbing around the ported engine: picks the manifest row, chooses eligible pages, hands the engine each page's staged name / DOB / ID, runs `verify_record`, persists page rows and the summary, writes three CSVs including the V1-shaped comparison file. |
| `lib/dos/stage.py` | **Stage 9.** Resolves the staged dates across the chart (`resolve.py`: progress-note spans, default-date pages), persists the primary pair plus every date, writes the DOS CSV. |
| `lib/ocr/reuse.py` | skip_ocr: reuse on-disk OCR artifacts instead of re-running OCR stages. |
| `lib/page_classify/stage.py` | Page type / codeability (`page_subtype`). |
| `lib/encounter/stage.py` | Encounter type (`encounter_type`). |
| `lib/sequencing/stage.py` | Page sequencing (`page_sequencing`). |
| `__init__.py` | Package marker. |

### `core-pipeline/stages/lib/` — layout

One folder per pipeline concern; only `canon_store.py` sits at the top level.

| Folder | Holds |
|---|---|
| `image_preprocess/` | Stage 1: rotation / tilt / mirror, handwriting classifier, quality score |
| `ocr/` | Docling final1 engine + OCR-time section-header pre-filter |
| `blank_junk/` | Blank / junk rules, TF-IDF model bridge, bundled model code (`model/`) |
| `page_classify/` | Page type / codeability (keyword families, header-weighted, family spans) |
| `encounter/` | Encounter type per visit (tiered evidence: page type → explicit setting text → hints) |
| `extraction/` | Key/value extraction (stage 6): the extractors, trained-model selection, staging, training code |
| `dos/` | Date-of-service resolution over the staged dates |
| `member/` | Member verification engine (rules over the staged name / DOB / ID) |
| `sequencing/` | Page sequencing |
| `keyword-canon/` | Every editable keyword JSON, reloaded on change (see `canon_store.py`) |

### `core-pipeline/stages/lib/image_preprocess/` — rotation + handwriting + quality

Live code for stage 1, not reference material.

| File | Role |
|---|---|
| `rotation.py` | `PageOrientationDetector` — coarse rotation, mirror and tilt detection from OpenCV/NumPy alone, plus `correct()`. |
| `osd.py` | Coarse page orientation from Tesseract OSD. |
| `hw_printed.py` | ConvNeXt printed vs handwritten when `.pth` is present. |
| `hw_printed_rf.py` | RandomForest fallback classifier. |
| `quality_analyzer.py` | Measured quality score / tag / warnings. Reads embedded DPI and scores it; does **not** resample or correct DPI. |
| `quality_label_postprocess.py` | Handwritten + High → Medium. |

### `core-pipeline/stages/lib/ocr/` — final1 engine + section headers

| File | Role |
|---|---|
| `docling_ocr.py` | Docling + RapidOCR `.pth` converter for final1. |
| `section_header_match.py` | Lexical (+ optional MiniLM) match against `keyword-canon/section_header_canon.json`. |
| `section_headers_io.py` | OCR-time header candidate filtering for final1/final2 JSON; the `section_headers` the pipeline keeps are written by `kv_extract`. |

Weight files live under **`core-pipeline/models/`** (gitignored), not under
`stages/`:

| Path | Purpose |
|---|---|
| `models/hw/handwritten_printed_convnext_tiny.pth` | ConvNeXt HW (stage 1, preferred) |
| `models/hw/handwritten_printed_convnext_tiny_backup.pth` | prior ConvNeXt HW fallback |
| `models/hw/image_type_classification.pkl` | RF HW fallback |
| `models/rapidocr/*.pth` + `ppocrv6_dict.txt` | Docling final1 |

### `core-pipeline/stages/lib/blank_junk/` — blank/junk classifier

Ported from `advantmed-imaging-ui/02-imaging-pipeline/junk-classification/`.

| File | Role |
|---|---|
| `classify.py` | The entry point: `classify_text()` tries each detector in priority order and returns a code; also the code constants, labels, `text_similarity()` / `fingerprint()` helpers and confidences. |
| `model_bridge.py` | `classify_page()`: the TF-IDF model decides keep / blank / junk; `classify_text()` names the junk subtype; regex fallback stamped `regex_fallback:<why>`. |
| `model/src/` | Bundled model code the pickled checkpoint (`models/blank-junk/tfidf_flat.joblib`) unpickles against. |
| `kw.py` | Compiles `keyword-canon/junk_keywords_canon.json`; reloads on edit. |
| `classify_junk.py` | The fuller CLI-era classifier retained from the V1 prototype. |
| `blank.py` | Blank detection: empty OCR, declared-blank phrasing, near-empty image. |
| `invoice.py`, `cover.py`, `record_request.py`, `instructions.py`, `letter_fax.py` | One junk category each. |
| `others.py` | Catch-all: gibberish OCR, signature-only pages. |
| `text_utils.py` | Shared text predicates — word count, gibberish, signature page. |
| `requirements.txt` | Dependency list inherited from the V1 prototype (the modules are pure stdlib). |
| `__init__.py` | Package marker. |

### `core-pipeline/stages/lib/member/` — member verification

Ported from the V1 `Member_Verification/` tree. **The verification rules are verbatim.**
The page's name, DOB and ID are no longer searched for here: the key/value extraction
(stage 6) stages them and the engine checks them against the manifest.

| File | Role |
|---|---|
| `engine.py` | The port of `run.py`'s decision flow: `staged_page_fields` (the staged name / DOB / ID vs the manifest), `verify_page`, `classify_page`, `document_verified`, and `verify_record` which drives a whole chart. Also `expected_from_manifest`, `detect_name_mode`, `summary_status`. |
| `__init__.py` | Public surface for the stage. |
| **`rules/`** | |
| `base_rules.py` | `combine_evidences` — the name+DOB/ID acceptance rule — and `is_present` (treats `"N/A"` as absent). |
| `name_2_words_rules.py` | Two-word verification: full match, or initial-only which needs both corroborators. |
| `name_3_words_rules.py` | Three-word verification: all three or two of three. |
| `wrong_member_rules.py` | `wrong_member_on_page` — true when the extraction found member names on the page and none is the expected member. The only route to a `Reject`. |
| `name_common.py` | The heart of name matching: tokenising, ignore/label/non-name vocabularies, `classify_two_word_name` / `classify_three_word_name` (including the `ONE_FULL_WRONG` "different member" case). |
| `field_match.py` | Does an extracted DOB / member ID say what the manifest says (the V1 comparisons, plus month-name dates). |
| `what_if_rules.py` | Page buckets (`Verified` / `Wrong_Member` / `Not_Verified`), `reject_threshold` = `min(5, ceil(10%))`, and `apply_what_if` → Accept/Reject. |
| `__init__.py` | Re-exports the rule surface. |
### `core-pipeline/stages/lib/extraction/` — key/value extraction

Built and trained as a standalone tool, integrated here as stage 6 (`kv_extract`). Full
description: [EXTRACTION.md](EXTRACTION.md).

| File / folder | Role |
|---|---|
| `stage.py` | The stage: reads Final2 word boxes, extracts the chart, stages it, writes headings into `section_headers`; `ensure_staging` re-runs it for a later stage when the staging is gone. |
| `engine.py` | `extract_chart` (extract, then let the trained version pick) and `readiness` (packages + weights, loads nothing). |
| `pipeline.py` | One pass per page over every extractor, then the headings. |
| `ocr_input.py` | Final2 page → the word-box page the extractors take; pages without boxes are reported. |
| `staging.py` | The per-chart staging file: write / read / drop, and the `Staged` / `StagedPage` views. |
| `dos/`, `member_dob/`, `member_id/`, `member_name/`, `provider_name/`, `electronic_signature/`, `page_no/`, `heading/` | One extractor each: `extract.py` finds candidates, `output.py` shapes rows, `keys.json` etc. are its catalogs. |
| `util/` | Geometry, dates, key catalog, GLiNER loader, model setup, settings (`config.py`). |
| `training/` | Dataset, training, evaluation and the registry of trained versions. |

### `core-pipeline/stages/lib/dos/` — date of service

The dates are found by the extraction; this folder decides which encounter each page belongs to.

| File | Role |
|---|---|
| `resolve.py` | `page_dates` (a page's staged dates; admit + discharge are one range) and `resolve_chart` (progress-note spans, non-encounter pages, default-date pages), driven by `dos_canon.json`. |
| `stage.py` | Reads the staging, resolves, writes the DB rows and the DOS CSV. |
| `__init__.py` | Package marker. |

### `review-ui/backend/app/`

| File | Role |
|---|---|
| `main.py` | FastAPI app: CORS, router mount, static frontend serving. |
| `api/routes.py` | Every viewer endpoint: health, config, folder list and detail, page images (local and blob-proxied), OCR text by kind, imaging results, CSV export. All `GET`. |
| `core/config.py` | Pydantic settings: `DATA_MODE`, data roots, database URL, blob viewer configuration, CORS. Derives `mode_label` for the UI pill. |
| `core/schemas.py` | Response models — `FolderSummary`, `FolderDetail`, `ImagingDocumentResponse`, `OcrTextResponse` — the contract the frontend types against. |
| `adapters/base.py` | The `FolderRepository` interface both modes implement. |
| `adapters/factory.py` | Chooses the adapter from `DATA_MODE`. |
| `adapters/local/repository.py` | **Local Mode.** Reads charts from `data/folders`: pages, OCR files, per-chart imaging CSVs with the legacy-pack fallback. Owns the scan cache and its mtime-based invalidation, and the per-chart stream marking that drives the folder-list badges. |
| `adapters/postgres/repository.py` | **Production Mode.** The same interface over the v8 tables. Maps `status` + `current_stage` to the UI pill, reads blank/junk from `v_page_blank_junk_final`, looks manifests up by `record_id`. **Page images** come from Azure Blob via `chart_list.blob_container` + `blob_path` (Raw_Input ingest order); Processed `output_path` is a fallback. `DATA_ROOT` is optional. |
| `services/imaging_overlays.py` | The CSV → UI field mapping: per-result `index_*_rows` functions, the column aliases each accepts, the canonical page-type labels, date formatting, and `collect_rows`' per-chart-then-legacy precedence. |
| `services/imaging_csv.py` | Streams the combined export CSV. |
| `services/metadata_csv.py` | Reads manifest CSVs for Local Mode. |
| `services/blob_store.py` | Proxies page images from Azure Blob (Entra or SAS). |
| `requirements.txt`, `Dockerfile` | Dependencies and runtime image. |
| `__init__.py` ×5 | Package markers. |

### `review-ui/frontend/src/`

| File | Role |
|---|---|
| `main.tsx` | React entry point. |
| `App.tsx` | Root: routing between landing, folder viewer and file viewer; auth gate; mode pill. |
| `LandingPage.tsx` | Chart list with search, status filters and stage badges. |
| `FolderViewer.tsx` | The main review screen — page image beside OCR text and imaging results. |
| `FileViewer.tsx` | Single-file browsing mode. |
| `ImagingPanel.tsx` | Renders the imaging results for the current page: blank/junk, page type, rotation, handwriting, member, DOS. |
| `FullscreenPageChrome.tsx` | Fullscreen page-viewing controls. |
| `PageJump.tsx` | Jump-to-page control. |
| `LoginPage.tsx` | Login screen. Shared POC pair **`imaging-user` / `aipocpw2026`**, overridable at build time with `VITE_LOGIN_USERNAME` / `VITE_LOGIN_PASSWORD`. **Credentials are inlined into the JS bundle and the backend has no auth**, so this gate protects nothing — see [Known limits](#6-known-limits). |
| `UserProfileMenu.tsx` | Profile/sign-out menu. |
| `BlobAuthModal.tsx` | Collects a SAS token for direct blob viewing. |
| `api.ts` | Typed client for the backend endpoints. |
| `blobAuth.ts` | SAS token normalising and storage. |
| `ocrPages.ts` | Splits combined OCR text on the `===== page =====` marker — the frontend end of that contract. |
| `ocrMatchRate.ts` | Compares OCR variants for the match-rate indicator. |
| `useImagePan.ts` | Pan/zoom hook for the page image. |
| `usePageViewerHotkeys.ts` | Keyboard navigation. |
| `styles.css` | Application styles. |
| `vite-env.d.ts` | Vite type shims. |

### `tests/`

| File | Role |
|---|---|
| `conftest.py` | Puts `core-pipeline`, `stages/lib` and the review-ui backend on `sys.path`, the way the services import them. |
| `requirements.txt` | What the suite needs — more than pytest, because the tests import the real modules, but not the OCR/ML stack, which is imported lazily. |
| `test_member_verification.py` | 52 tests pinning the ported engine against the reference: evidence combination, two/three-word classification, DOB and MemberID extraction, the three page buckets, the reject threshold formula, and end-to-end record verification. |
| `test_chart_status.py` | Status derivation: earliest-incomplete-stage, the two blank/junk passes as distinct stages, skipped-counts-as-done, failure escalation, terminal states. |
| `test_contracts.py` | The core-pipeline ↔ review-ui seam: every column the UI reads is a column a stage writes, the junk subtype vocabulary matches the schema CHECK, every per-chart CSV suffix is one the UI branches on, the OCR marker round-trips, and `STAGE_CHAIN` matches the `pipeline_stage` seed. |
| `test_blank_junk_and_dos.py` | Duplicate scoping across passes, subtype mapping, DOS multi-date rows, and that the DOS driver really does carry forward and emit ISO. |

---

## 5. Design decisions

**Stage order lives in a table, not in code.** `pipeline_stage` drives progress
reporting and status derivation; `STAGE_CHAIN` drives execution. A contract test
keeps them in step. Adding a stage is an INSERT plus a module.

**Progress is rows, not columns.** v6's 11 `page_list.*_status` columns meant a
CHECK migration per stage and could not represent one stage running twice —
which is exactly what blank/junk does. `page_stage_status` keyed
`(page_id, stage_name, pass_no)` fixes both, and gives resume its query.

**Resume is the default; `force` is explicit.** Stage 5 is billed per page. A
default that re-ran everything would make re-running expensive enough to avoid,
which is the wrong incentive when charts fail part-way.

**Degradation is recorded, never silent.** A page the extraction could not read
(no word boxes) is skipped as `no_word_boxes` and logged; the later stages see it as a
page with nothing found. A missing extraction model fails the stage rather than
changing what every later stage reads. Both appear in `GET /health`. A missing
capability must be visible in the data, not inferable only from a log line.

**Files are rebuilt, not appended.** Every CSV and combined text file is
regenerated from the database. v6 truncated in one pass and appended in the next,
so re-running a pass duplicated rows.

**The reference is the source of truth.** Ported logic is verbatim where it can
be; adaptations are commented with what changed and why. The tests pin the
reference's behaviour, and `_member_v1_compare.csv` is written in the
reference's own column order so a run can be diffed directly against V1.

---

## 6. Known limits

Things a reader should know before relying on this in production.

**No authentication anywhere.** `LoginPage.tsx` checks a credential pair
compiled into the client bundle — `imaging-user` / `aipocpw2026`, committed to
this repository on purpose because it is shared with every POC user and secures
nothing — and the review-ui backend has no auth on any
route — page images, OCR text and the full CSV export are open to anyone who can
reach the port. core-pipeline's API is equally open. For charts carrying member
names and dates of birth this needs a real answer before any non-private
network. Not addressed here; it is not a viewer change.

**The review UI cannot record a review.** Every route is a `GET`. `manual_review`,
`rejection_results` and `audit_log` exist in the schema and nothing writes them.
Scoped out deliberately — the UI stays a viewer.

**No work queue.** Each ingest runs its chain in the API process via a
FastAPI background task. `pipeline_jobs` carries `lease_expires_at`,
`heartbeat_at` and `attempt` so it *can* back a claim/lease worker, but no
worker exists. Consequences: no concurrency cap across charts, no automatic
retry of a stage that raised, and work in flight is lost if the process restarts
(though resume means a re-run picks up where it stopped).

**Parallelism is within a stage, not across charts.** `STAGE_WORKERS` threads
pages inside one stage. Two charts ingested at once run two full chains in one
process.

**The extraction weights are not all in git.** The trained ranker (`models/kv_ranker/v002`, ~1 MB) is
committed; GLiNER (~585 MB) and the Heron detector (~164 MB) are copied into `models/` or fetched with
`python -m stages.lib.extraction.util.model_setup`. `GET /health` → `extraction` names what is missing.

**The SQL is syntax-validated, not run.** `v1.sql` and `v2.sql` parse
clean under a real PostgreSQL parser (`pglast`), and the code paths that use them
are unit-tested — but no PostgreSQL server was available in this environment, so
neither file has been executed. Apply migration 002 to a restorable snapshot
first.

**`review-ui/` has forked from `05-imaging-ui/`.** They share ancestry and have
diverged. Treat `05-imaging-ui` as frozen; changes belong here.
