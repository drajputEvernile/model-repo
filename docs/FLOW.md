# Flow — what happens when

What moves through the system, in order, and what each step leaves behind.
For *why* each step decides what it decides, see [LOGIC.md](LOGIC.md).
For the HTTP surface, see [API.md](API.md).

---

## 1. The two deployments

```mermaid
flowchart LR
  subgraph CP["core-pipeline (own docker-compose, port 8001)"]
    API["FastAPI<br/>ingest · rerun · manifest"]
    ORCH["Orchestrator<br/>9-stage chain"]
    API --> ORCH
  end

  subgraph RU["review-ui (own docker-compose, Docker ports 4000/4001)"]
    RAPI["FastAPI viewer :4000 (host) / :3000 (container)"]
    WEB["Vite + React :4001 (host) / :3001 (container)"]
    WEB --> RAPI
  end

  BLOB[("Azure Blob<br/>chart folders + manifests")]
  PG[("PostgreSQL<br/>schema v8")]
  VOL[/"data/folders<br/>shared volume"/]

  BLOB -- "download" --> ORCH
  ORCH -- "writes" --> PG
  ORCH -- "writes" --> VOL
  RAPI -- "reads" --> PG
  VOL -- "read-only" --> RAPI

  style CP fill:#e8f0fe,stroke:#4a76c7
  style RU fill:#eaf6ec,stroke:#4c9a5b
```

The two services **never call each other**. They share the database and the
`data/folders` volume — core-pipeline writes it, review-ui mounts it read-only.
Either can be redeployed without the other.

---

## 2. End-to-end chart flow

```mermaid
flowchart TD
  START(["POST /api/charts/run<br/>{input_type, container_name, input_path, chart_name}"]) --> DL

  subgraph INTAKE["Intake"]
    DL["Download pages<br/>→ data/folders/&lt;chart&gt;/pages/1.jpg…N"]
    DL --> REG["chart_list + page_list<br/>sha256 + size per page"]
    REG --> SEED["Seed page_stage_status<br/>(pending × pages × stages)"]
    SEED --> LINK["Link manifest rows<br/>swept earlier"]
  end

  LINK --> S1

  subgraph CHAIN["Stage chain — API reprocesses every page; CLI --resume skips completed pages"]
    S1["1 · ocr_quality<br/>rotation + handwriting"]
    S2["2 · ocr_prelim<br/>Tesseract, every page"]
    S3["3 · blank_junk pass 1<br/>printed pages, prelim text"]
    S4["4 · ocr_final1<br/>Docling+RapidOCR"]
    S5["5 · ocr_final2<br/>Azure DocIntel · billed"]
    S5b["6 · kv_extract<br/>member · DOS · page no · headings → staging"]
    S6["7 · blank_junk pass 2<br/>handwritten + survivors"]
    S7["8 · member_verify<br/>staged fields → rules → what-if"]
    S8["9 · dos_extract<br/>staged dates → encounter spans"]
    S9["10 · page_subtype<br/>TF codeable / continue-until-DOS"]
    S10["11 · encounter_type<br/>F2F / Tele / IP / Home per DOS"]
    S11["12 · page_sequencing<br/>markers → streams → suggested order"]
    S1 --> S2 --> S3 --> S4 --> S5 --> S5b --> S6 --> S7 --> S8 --> S9 --> S10 --> S11
  end

  S11 --> DONE["refresh_chart_status<br/>→ completed / needs_review / failed"]

  style S5 fill:#fde8e8,stroke:#c74a4a
  style S7 fill:#fff4e0,stroke:#c78a4a
```

Stages 5 and 7 are highlighted: **stage 5 costs money per page** (which is why
`skip_ocr` matters), and **stage 7 produces the accept/reject decision**.

---

## 3. What each stage reads and writes

| # | Stage | Runs on | Reads | Writes to Postgres | Writes to disk |
|---|-------|---------|-------|--------------------|----------------|
| 1 | `ocr_quality` | every page | page image | `ocr_quality_results` | `imaging/<chart>_rotation.csv`, `_hw_printed.csv`, `_quality.csv`, and `corrected-pages/<n>.jpg` when `ROTATION_CORRECTION_ENABLED` |
| 2 | `ocr_prelim` | every page | corrected page if one exists, else page image | `ocr_results` (`tesseract`) | `ocr/<chart>_prelim.txt` |
| 3 | `blank_junk` pass 1 | printed + non-low-quality only | prelim text | `blank_junk_classification` (pass 1) | `imaging/<chart>_junk.csv` |
| 4 | `ocr_final1` | not blank/junk, + HW / low-quality | corrected page if one exists, else page image | `ocr_results` (`docling`) | `ocr/<chart>_final1.json` (incl. `section_header_candidates`) |
| 5 | `ocr_final2` | not blank/junk, + HW / low-quality; **skips high-quality printed** | corrected page if one exists, else page image | `ocr_results` (`azuredocintel`) | `ocr/<chart>_final2.json` (incl. candidates / `pagesMeta`) |
| 6 | `kv_extract` | pages with Final2 word boxes (see [EXTRACTION.md](EXTRACTION.md)) | `ocr_results` (`azuredocintel`) + the page image | headings into the `ocr_results` JSON blobs | `staging/extraction.json` (dropped when the chart completes); `section_headers` in `*_final1.json` / `*_final2.json` |
| 7 | `blank_junk` pass 2 | HW + low-quality + surviving printed | final2 text, else final1 | `blank_junk_classification` (pass 2), then `is_final` stamped | rewrites `_junk.csv` |
| 8 | `member_verify` | not blank/junk/duplicate | staged name / DOB / ID + `manifest_member_list` | `member_extraction_results`, `member_verification_summary` | `_member_extraction.csv`, `_member_verification.csv`, `_member_v1_compare.csv` |
| 9 | `dos_extract` | not blank/junk/duplicate | staged dates + best text (page type) | `dos_extraction_results` | `imaging/<chart>_dos.csv` |
| 10 | `page_subtype` | every page | best text + DOS + blank/junk | `page_classification` (main = TF; blank/junk/dup = `non_codeable` + junk subtype) | `imaging/<chart>_codeable.csv` |
| 11 | `encounter_type` | not blank/junk/duplicate | best text + DOS | `encounter_type_results` | `imaging/<chart>_encounter.csv` |
| 12 | `page_sequencing` | all pages (junk flagged) | best text + staged printed page numbers | `page_sequencing_results` | `imaging/<chart>_sequencing.csv` |

Every stage also writes one `pipeline_jobs` row and updates
`page_stage_status` per page.

**"Best text"** means final2 → final1 → prelim (restricted). Prelim is never used
for handwritten / uncertain / mixed / low-quality pages.

---

## 4. Skip rules — why a page drops out

```mermaid
flowchart TD
  P["Page"] --> Q{"HW / uncertain / mixed<br/>or quality_tag=low?"}
  Q -- yes --> HW["Skip blank/junk pass 1"]
  Q -- no --> BJ1["Blank/junk pass 1<br/>on prelim text"]

  HW --> F1["Final OCR 1"]
  BJ1 --> D{"blank / junk /<br/>duplicate?"}
  D -- yes --> STOP1(["Skipped: no final OCR,<br/>no member, no DOS"])
  D -- no --> F1

  F1 --> HQ{"high quality<br/>+ printed?"}
  HQ -- yes --> BJ2["Blank/junk pass 2<br/>(final1 text; Azure skipped)"]
  HQ -- no --> F2["Final OCR 2 (Azure)"]
  F2 --> BJ2b["Blank/junk pass 2<br/>final2 else final1"]
  BJ2 --> D2{"blank / junk /<br/>duplicate?"}
  BJ2b --> D2
  D2 -- yes --> STOP2(["Skipped: no member, no DOS"])
  D2 -- no --> EXTRACT["member_verify + dos_extract<br/>best text: f2→f1→prelim*"]
```

\* prelim only for printed non-low-quality pages.

  style STOP1 fill:#f5f5f5,stroke:#999
  style STOP2 fill:#f5f5f5,stroke:#999
```

Each skip is recorded — `page_stage_status.status='skipped'` with a
`skip_reason` (`handwritten`, `blank_junk_pass1`, `blank_junk`,
`no_final2_text`). A skipped page counts as *done* for chart-status purposes,
so a chart of blank pages still reaches `completed`.

### Adaptive skip_ocr (`skip_ocr=true`)

When OCR artifacts already exist, the orchestrator:

1. Hydrates `ocr/` / `ocr_results` (same as plain skip_ocr).
2. **Force-refreshes** `ocr_quality` (rotation + HW + measured quality).
3. Compares each page’s gate signature
   `(hw_class, quality_tag, rotation_applied, orientation_bucket)` to the
   pre-run snapshot (`core-pipeline/stages/utilities/gate_delta.py`).
4. Sets OCR-related `page_stage_status` rows back to pending only when
   artifacts are missing or the gate path changed (so OCR engines may re-run
   for those pages).
5. **Force-re-runs every non-OCR stage** — blank/junk (both passes), section
   headers, member, DOS, codeable, encounter, sequencing — regardless of
   prior completion.

```bash
curl -X POST localhost:8001/api/charts/run -H 'Content-Type: application/json' \
  -d '{"input_type": "blob", "container_name": "imaging-pipeline",
       "input_path": "Raw_Input/Run1/Batch1/DEID_PNGs",
       "chart_name": "52754737_48221214", "skip_ocr": true}'
```

---

## 5. Manifest flow — runs independently

```mermaid
flowchart LR
  M1[/"metadata_R1_B1.csv<br/>local file, directory, or blob prefix"/]
  M1 --> SWEEP["POST /api/manifest/sweep"]
  SWEEP --> PARSE["Parse rows<br/>split first / middle / last"]
  PARSE --> UP["Upsert manifest_member_list<br/>keyed on record_id"]
  UP --> LINKED{"Chart already<br/>ingested?"}
  LINKED -- yes --> SETID["chart_id set now"]
  LINKED -- no --> WAIT["chart_id stays NULL —<br/>ingest links it later"]

  ING["Chart ingest"] --> LINK2["link_manifest_to_chart(record_id)"]
  WAIT -.-> LINK2
```

The manifest is a fact about a **client RecordId**, not about a chart row we
happen to hold. Sweeping does **not** create placeholder charts — v6 did, which
filled the review UI with empty charts for records never ingested.

Order does not matter: sweep before ingest or after, the link is made either
way. `run_id` / `batch_id` come from the `R#`/`B#` in the filename unless
overridden.

---

## 6. Resume — what a re-run actually does

```mermaid
sequenceDiagram
  participant C as Caller
  participant O as Orchestrator
  participant DB as page_stage_status
  participant AZ as Azure DocIntel

  Note over O,AZ: First run — dies at page 401 of 500
  C->>O: POST /api/charts/run
  O->>AZ: analyse pages 1…400
  AZ-->>DB: 400 rows 'completed'
  O--xO: crash

  Note over O,AZ: API re-run — same input_path + chart_name, reprocesses all
  C->>O: POST /api/charts/run {"input_path":…,"chart_name":…}
  O->>AZ: analyse pages 1…500 again
  Note right of AZ: billed again

  Note over O,AZ: CLI resume (python cli.py run --resume)
  C->>O: cli.py run --chart-id … --resume
  O->>DB: pages_needing_stage('ocr_final2')
  DB-->>O: pages 401…500 only
  O->>AZ: analyse 100 pages
```

- API re-run (same `input_path` + `chart_name`) → workspace pages are reused
  (`skip_page_download` defaults to `true`) and every stage **reprocesses**.
  There is no `force` field and no page-level resume on the API.
- `{"skip_ocr": true}` → reuse existing OCR; no Final2 re-billing (see §4).
- `{"skip_completed": true}` → do not run a chart whose status is already
  `completed` / `needs_review` / `rejected`.
- `{"only": ["member_verify"]}` → run just that stage.
- CLI `--resume` → completed and skipped pages are left alone. The CLI keeps
  its own flags; they are not the API's fields.

---

## 7. Chart status over time

`chart_list.status` is lifecycle; `current_stage` is position. Both are derived
after every stage from the page rows — never set by hand.

```mermaid
stateDiagram-v2
  [*] --> received: chart row created
  received --> downloading: intake starts
  downloading --> processing: pages registered
  processing --> processing: stage completes,<br/>current_stage advances
  processing --> failed: a page failed in<br/>an incomplete stage
  processing --> completed: all stages done
  processing --> needs_review: done, but member<br/>verification unresolved
  failed --> processing: rerun
```

Accept/reject lives on `member_verification_summary`, not on `chart_list.status`
(``rejected`` is legacy and is no longer written).

**The rule:** `current_stage` is the earliest stage, in `pipeline_stage.seq`
order, where not every page is `completed` or `skipped`.

Because stage order lives in a table rather than in code, registering
page-subtype / encounter / sequencing later is an `INSERT` plus a stage module —
no migration, no change to this logic.

---

## 8. Timeline of one 500-page chart

```mermaid
gantt
  title Where the wall-clock goes (indicative, STAGE_WORKERS=4)
  dateFormat X
  axisFormat %s
  section Intake
  Download 500 images        :0, 60
  section OCR
  Tesseract prelim           :60, 180
  Rotation + handwriting     :240, 120
  section Triage
  Blank/junk pass 1          :360, 10
  section Final OCR
  RapidOCR (final1)          :370, 300
  Azure DocIntel (final2)    :670, 400
  section Triage
  Blank/junk pass 2          :1070, 10
  section Extraction
  Member verify              :1080, 40
  DOS extract                :1120, 60
```

Blank/junk pass 1 pays for itself: every page it rules out is a page the two
final OCR stages never touch, and stage 5 is the billed one.
