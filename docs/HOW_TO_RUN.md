# How to Run (and re-run without OCR)

Practical recipes for charts that **already finished OCR** (e.g. hundreds of
docs with `ocr/` + `ocr_results` in the workspace) when you only want later
stages — codeable, encounter, sequencing, DOS, member, section headers — or a
selective re-pass.

All examples below use **Azure Blob** paths. Local paths work the same way:
set `"input_type": "local"`, drop `container_name`, and make `input_path` /
`output_path` directories on the server (see [`API.md`](API.md)).

Full API reference: [`API.md`](API.md). Stage flow: [`FLOW.md`](FLOW.md).

**Request field cheat-sheet** (`POST /api/charts/run` and `/batch-run`)

| Field | Run | Batch | Meaning | Example |
|---|---|---|---|---|
| `input_type` | required | required | `"local"` or `"blob"` | `"blob"` |
| `container_name` | blob only | blob only | Storage container, used for read **and** write. Omit for local | `"imaging-pipeline"` |
| `input_path` | required | required | Folder / prefix that holds the chart folders | `"Raw_Input/Run1/Batch1/DEID_PNGs"` |
| `output_path` | optional | optional | Results go to `<output_path>/<chart_name>/`, replacing existing files. Omit = no write | `"Processed/Run1"` |
| `chart_name` | required | — | One chart folder under `input_path` (= chart name) | `"52743839_44976074"` |
| `chart_list` | — | optional | Only these chart folders (list or comma-separated string). Not found → `charts_missing` | `["52743839_44976074"]` |
| `sample` | — | optional | Run at most N charts, unfinished ones first | `10` |

Batch-run takes the same container + input/output paths **without**
`chart_name` (each subfolder under `input_path` is one chart). Any field not
listed here or in §2 is rejected with **422** — including the old
`blob_container` / `blob_read_path` / `blob_read_folder_name` /
`blob_write_path` / `local_*` names, `run_id` / `batch_id` (now always inferred
from the path: `Run1/Batch1` → `R1`/`B1`) and `workers` (set `BATCH_WORKERS`
in `core-pipeline/.env`; `BATCH_WORKERS × STAGE_WORKERS + 2 ≤ DB_POOL_MAX` or
batch-run returns 400).

The CLI (`python cli.py run` / `batch-run`) keeps its own flags (`--resume`,
`--through`, `--workers`, `--test-mode`, …); the CLI examples below are
unchanged.

---

## 0. One-time: bring the DB up to date

If these charts ran on an **older** schema (before codeable / encounter /
sequencing), apply the additive patch first — do **not** re-run `v1.sql` on a
live DB:

```bash
psql "$DATABASE_URL" -f schema/patch_output_path.sql
```

That creates `encounter_type_results` + `page_sequencing_results`, registers
the three new stages, and seeds `page_stage_status` pending rows.

First-time empty database instead:

```bash
psql "$DATABASE_URL" -f schema/v1.sql
# optional: psql "$DATABASE_URL" -f schema/v2.sql
```

Confirm stages (and that blob credentials are live):

```bash
curl -fsS localhost:8001/ready
curl -fsS localhost:8001/health | python -m json.tool   # blob.ready should be true
curl -fsS localhost:8001/api/stages | python -m json.tool
```

---

## 1. Stage names (chain order)

| Token | What it does | Bills Azure DI? |
|---|---|---|
| `ocr_quality` | Rotation + handwriting | No |
| `ocr_prelim` | Tesseract | No |
| `blank_junk` / `blank_junk:2` | Junk pass 1 / pass 2 | No |
| `ocr_final1` | RapidOCR / Docling | No |
| `ocr_final2` | Azure Document Intelligence | **Yes — per page** |
| `kv_extract` | Member / DOS / page number / headings from the OCR word boxes — see [EXTRACTION.md](EXTRACTION.md) | No |
| `member_verify` | Member extract + verify | No |
| `dos_extract` | Date of service | No |
| `page_subtype` | Codeable / Non Codeable / Discharge Frequency | No |
| `encounter_type` | Outpatient F2F / Tele / Inpatient / Home | No |
| `page_sequencing` | Suggested page order | No |

---

## 2. The knobs

Every API run **reprocesses** every stage it runs — there is no `force` field
any more, and no page-level resume on the API. `skip_ocr` is the only thing
that avoids re-running OCR (and re-billing Final2). Outputs always replace
what is at `output_path` (no `overwrite` / `write_mode`; original pages are
never written back).

| Field | Default | Use when |
|---|---|---|
| `only: ["stage", …]` | omit (whole chain) | Run **just** those stages, against what is on disk (quality still runs under `skip_ocr`). |
| `run_through: "stage"` | omit (end of chain) | Run from the top of the chain and **stop after** that stage. (Was `through`.) |
| `skip_ocr: true` | `false` | Reuse existing OCR instead of re-running the OCR engines — no Final2 billing. Quality/rotation and every non-OCR stage still re-run. See § skip_ocr below. |
| `skip_completed: true` | `false` | Skip charts whose `chart_list.status` is `completed` / `needs_review` / `rejected`. Run returns **200** `{"status": "skipped"}`; batch lists them in `charts_skipped_completed`. |
| `skip_page_download` | **`true`** | Reuse page images already in the workspace. `false` = wipe `pages/` + `corrected-pages/` and re-fetch from `input_path`. (Replaces `redownload_pages: true`.) |
| `--test-mode` (CLI only) | off | **Local only.** No Postgres. Workspace under `data/folders/<chart>-test` for review-ui. Env `TEST_MODE=true`. Not available on the API. |

Re-running a chart with the same `input_path` + `chart_name` resumes it:
workspace pages are reused and every stage reprocesses. There is no `chart_id`
resume on the API.

### What `--test-mode` does (CLI)

For laptop / folder experiments when Postgres is unavailable — or when you
want a side-by-side folder you can open in review-ui without touching the
real chart:

```bash
cd core-pipeline && source .venv/bin/activate
python cli.py run \
  --local-read-path /data/inbox \
  --folder-name 52743839_44976074 \
  --test-mode
```

- Reads images from `/data/inbox/52743839_44976074/`
- Writes workspace to `data/folders/52743839_44976074-test/` (pages, ocr, imaging)
- Never opens Postgres (in-memory stage state for this process only)
- Rejects blob sources and `--chart-id` / `--chart-name` resume
- review-ui Local Mode: open the `…-test` folder to inspect
- Member verify: if MemoryStore has no manifesto row, loads matching
  `recordId` from on-disk CSVs under `review-ui/data/metadata/` or
  `review-ui/data/folders/manifest/` (strips the `-test` suffix for the lookup)

`--skip-db-write` is a deprecated alias for the same behaviour.

### What `skip_ocr` does

Looks under `data/folders/<chart>/` (review-ui workspace):

1. **`pages/` present** → use it. **Missing** → download from **Raw_Input** (`input_path`).
2. **`ocr/` present** → use it. **Missing** → pull from **Processed** (`output_path`). **Still missing** → materialize from Postgres `ocr_results`. **Still missing** → **re-run OCR engines**.
3. **Quality + rotation always re-run** → rewrite `corrected-pages/`.
4. **Every non-OCR stage force-re-runs** — blank/junk (both passes), section headers, member, DOS, codeable, encounter, sequencing. Gate-delta may still reopen OCR engines for a page whose HW/quality/rotation path flipped or whose reused OCR is missing.
5. **Write** (if `output_path` is set) **replaces** destination `ocr/`, `corrected-pages/`, `imaging/` (original `pages/` are never written).

```bash
curl -X POST localhost:8001/api/charts/run -H 'Content-Type: application/json' \
  -d '{"input_type": "blob", "container_name": "imaging-pipeline",
       "input_path": "Raw_Input/Run1/Batch1/DEID_PNGs",
       "chart_name": "52743839_44976074",
       "output_path": "Processed/Run1", "skip_ocr": true}'
```

Rule of thumb for a large blob batch that already finished OCR:

- Want new classifiers only → **`only`** (workspace pages reused by default), optional `output_path` to sync CSVs out.
- Want “reuse OCR, re-run everything else” → **`skip_ocr: true`** (§4 / §6B).
- Never run a full chain (no `only`, no `skip_ocr`) unless you intend to pay for Final2 again — every API run reprocesses.

---

## 3. Fresh chart from blob (full pipeline)

```bash
curl -X POST localhost:8001/api/charts/run -H 'Content-Type: application/json' \
  -d '{
    "input_type": "blob",
    "container_name": "imaging-pipeline",
    "input_path": "Raw_Input/Run1/Batch1/DEID_PNGs",
    "chart_name": "52743839_44976074",
    "output_path": "Processed/Run1"
  }'
```

`run_id` / `batch_id` (`R1` / `B1`) come from `Run1/Batch1` in the path; they
are not request fields.

CLI:

```bash
cd core-pipeline && source .venv/bin/activate
python cli.py run \
  --blob-container imaging-pipeline \
  --blob-read-path Raw_Input/Run1/Batch1/DEID_Images \
  --blob-read-folder-name 52743839_44976074 \
  --blob-write-path Processed/Run1/Batch1 \
  --run-id R1 --batch-id B1
```

Stop before Azure OCR (no Final2 bill):

```bash
curl -X POST localhost:8001/api/charts/run -H 'Content-Type: application/json' \
  -d '{
    "input_type": "blob",
    "container_name": "imaging-pipeline",
    "input_path": "Raw_Input/Run1/Batch1/DEID_PNGs",
    "chart_name": "52743839_44976074",
    "output_path": "Processed/Run1",
    "run_through": "ocr_final1"
  }'
```

---

## 4. Your case: OCR already done — run new stages + refreshed quality

Charts already exist in Postgres and in the workspace. Send the same
`input_path` + `chart_name` as the original run — `skip_page_download`
defaults to `true`, so workspace pages are reused rather than re-downloaded.

### A. New stages only (codeable + encounter + sequencing) — **no OCR**

Safe when HW/quality is fine and you only need the new classifiers. An
`only` list without `ocr_final2` means Final2 is **not** re-billed:

```bash
curl -X POST localhost:8001/api/charts/run -H 'Content-Type: application/json' \
  -d '{
    "input_type": "blob",
    "container_name": "imaging-pipeline",
    "input_path": "Raw_Input/Run1/Batch1/DEID_PNGs",
    "chart_name": "52743839_44976074",
    "only": ["page_subtype", "encounter_type", "page_sequencing"],
    "output_path": "Processed/Run1"
  }'
```

Batch (sample of 10, or omit `sample` for the whole drop):

```bash
curl -X POST localhost:8001/api/charts/batch-run -H 'Content-Type: application/json' \
  -d '{
    "input_type": "blob",
    "container_name": "imaging-pipeline",
    "input_path": "Raw_Input/Run1/Batch1/DEID_PNGs",
    "output_path": "Processed/Run1",
    "sample": 10,
    "only": ["page_subtype", "encounter_type", "page_sequencing"]
  }'
```

Concurrency comes from `BATCH_WORKERS` in `core-pipeline/.env`, not the
request. The CLI below still takes `--workers`:

```bash
python cli.py batch-run \
  --blob-container imaging-pipeline \
  --blob-read-path Raw_Input/Run1/Batch1/DEID_PNGs \
  --blob-write-path Processed/Run1/Batch1 \
  --sample 10 --force \
  --only page_subtype --only encounter_type --only page_sequencing \
  --workers 2
```

Outputs: `imaging/<chart>_codeable.csv`, `*_encounter.csv`, `*_sequencing.csv`
(+ matching Postgres tables).

### B. Better printed/HW + quality model — **reuse OCR unless quality switches**

Deploy the new weights under `models/hw/`, then:

```bash
curl -X POST localhost:8001/api/charts/batch-run -H 'Content-Type: application/json' \
  -d '{
    "input_type": "blob",
    "container_name": "imaging-pipeline",
    "input_path": "Raw_Input/Run1/Batch1/DEID_PNGs",
    "output_path": "Processed/Run1",
    "skip_ocr": true,
    "only": [
      "ocr_quality",
      "page_subtype",
      "encounter_type",
      "page_sequencing"
    ]
  }'
```

What this does:

1. **`skip_ocr: true`** — reuse workspace / Processed / DB OCR.
2. **Quality always re-runs** (even if you omit `ocr_quality` from `only` when
   `skip_ocr` is on) → new HW/quality model writes fresh tags + `corrected-pages/`.
3. **Gate-delta** — OCR engines re-open **only** for pages whose
   printed/HW/quality/rotation path flipped; everyone else keeps existing OCR.
4. Then codeable / encounter / sequencing run on the refreshed quality +
   existing text.

If encounter needs fresh DOS first, add `"dos_extract"` to `only` (still no
Final2 unless a page is gate-reopened into `ocr_final2`).

### C. Also refresh DOS first (encounter often needs it)

```bash
curl -X POST localhost:8001/api/charts/run -H 'Content-Type: application/json' \
  -d '{
    "input_type": "blob",
    "container_name": "imaging-pipeline",
    "input_path": "Raw_Input/Run1/Batch1/DEID_PNGs",
    "chart_name": "52743839_44976074",
    "only": ["dos_extract", "page_subtype", "encounter_type", "page_sequencing"],
    "output_path": "Processed/Run1"
  }'
```

Poll: `GET /api/charts/by-name/{chart_name}` (blob) / `GET /api/charts/{id}` (local, id in the 202 body) or core-pipeline logs / `progress.txt` under the chart.

---

## 5. Re-run one component only

| Goal | `only` |
|---|---|
| Extraction + headings (after changing the model version) | `["kv_extract"]` |
| Blank/junk pass 2 | `["blank_junk:2"]` |
| Member verification (reads the staged extraction; re-runs it if the chart already completed) | `["member_verify"]` |
| DOS only | `["dos_extract"]` |
| Codeable only | `["page_subtype"]` |
| Encounter only | `["encounter_type"]` |
| Sequencing only | `["page_sequencing"]` |

```bash
curl -X POST localhost:8001/api/charts/run -H 'Content-Type: application/json' \
  -d '{
    "input_type": "blob",
    "container_name": "imaging-pipeline",
    "input_path": "Raw_Input/Run1/Batch1/DEID_PNGs",
    "chart_name": "52743839_44976074",
    "only": ["kv_extract"],
    "output_path": "Processed/Run1"
  }'
```

---

## 6. Skip OCR but re-run the rest of the chain

### A. Explicit post-OCR list (clearest — no Final2)

```bash
curl -X POST localhost:8001/api/charts/run -H 'Content-Type: application/json' \
  -d '{
    "input_type": "blob",
    "container_name": "imaging-pipeline",
    "input_path": "Raw_Input/Run1/Batch1/DEID_PNGs",
    "chart_name": "52743839_44976074",
    "only": [
      "kv_extract",
      "blank_junk:2",
      "member_verify",
      "dos_extract",
      "page_subtype",
      "encounter_type",
      "page_sequencing"
    ],
    "output_path": "Processed/Run1"
  }'
```

Batch equivalent:

```bash
curl -X POST localhost:8001/api/charts/batch-run -H 'Content-Type: application/json' \
  -d '{
    "input_type": "blob",
    "container_name": "imaging-pipeline",
    "input_path": "Raw_Input/Run1/Batch1/DEID_PNGs",
    "output_path": "Processed/Run1",
    "only": [
      "kv_extract",
      "blank_junk:2",
      "member_verify",
      "dos_extract",
      "page_subtype",
      "encounter_type",
      "page_sequencing"
    ]
  }'
```

### B. `skip_ocr` — reuse OCR, re-run everything else

Reuses OCR on disk; refreshes quality; **force-re-runs blank/junk and all
later stages**. OCR engines only re-run for pages gate-delta marks pending:

```bash
curl -X POST localhost:8001/api/charts/run -H 'Content-Type: application/json' \
  -d '{
    "input_type": "blob",
    "container_name": "imaging-pipeline",
    "input_path": "Raw_Input/Run1/Batch1/DEID_PNGs",
    "chart_name": "52743839_44976074",
    "skip_ocr": true,
    "output_path": "Processed/Run1"
  }'
```

Final2 may still bill for pages that leave the `high+printed` skip path or
lack Final2 text. Prefer §6A when you want **zero** OCR billing.

---

## 7. Resume a half-finished chart / batch

Page-level resume ("skip completed pages") is **CLI only** now — the API has
no `force: false`. Over the API, re-sending the same `input_path` +
`chart_name` reuses workspace pages but **reprocesses every stage** (Final2
bills again unless you pass `skip_ocr: true`).

CLI, skipping completed pages:

```bash
python cli.py run --chart-id 123 --resume \
  --blob-write-path Processed/Run1/Batch1
```

Batch after a timeout — **same `input_path`**. Over the API, skip charts that
already finished and reuse OCR for the rest:

```bash
curl -X POST localhost:8001/api/charts/batch-run -H 'Content-Type: application/json' \
  -d '{
    "input_type": "blob",
    "container_name": "imaging-pipeline",
    "input_path": "Raw_Input/Run1/Batch1/DEID_PNGs",
    "output_path": "Processed/Run1",
    "skip_completed": true,
    "skip_ocr": true
  }'
```

CLI equivalent with page-level resume:

```bash
python cli.py batch-run \
  --blob-container imaging-pipeline \
  --blob-read-path Raw_Input/Run1/Batch1/DEID_Images \
  --blob-write-path Processed/Run1/Batch1 \
  --resume --workers 2
```

---

## 8. What a re-run does to your wallet

| Call | Final2 (Azure DI) |
|---|---|
| `only` list with **no** `ocr_final2` | Not billed |
| Full chain / no `only`, no `skip_ocr` | **Re-bills every page** (the API always reprocesses) |
| `skip_completed: true` | Finished charts not run, so not billed |
| `skip_ocr: true` | Usually none; rare pages if gate-delta reopens them |
| CLI `--resume` | Bills only pages still pending Final2 |

---

## 9. Write path / review-ui

`output_path` writes results to `<output_path>/<chart_name>/`, replacing
existing files (same backend as `input_type`; omit it to run without writing). Workspace CSVs under each chart’s `imaging/` are what Local Mode
review-ui overlays (`*_codeable.csv`, `*_encounter.csv`, `*_sequencing.csv`, …).
Production Mode reads Postgres (`encounter_type_results`,
`page_sequencing_results`, …).

---

## 10. Quick checklist for “500 docs on blob, OCR done, add new classifiers”

1. Confirm `GET /ready` and `blob.ready` on `GET /health`
2. Deploy new HW/quality weights under `models/hw/` if refreshing quality
3. Batch — **new stages only** (no OCR bill):

```bash
curl -X POST localhost:8001/api/charts/batch-run -H 'Content-Type: application/json' \
  -d '{
    "input_type": "blob",
    "container_name": "imaging-pipeline",
    "input_path": "Raw_Input/Run1/Batch1/DEID_PNGs",
    "output_path": "Processed/Run1",
    "only": ["page_subtype", "encounter_type", "page_sequencing"]
  }'
```

   Or **refresh quality + reuse OCR** (§4B):

```bash
curl -X POST localhost:8001/api/charts/batch-run -H 'Content-Type: application/json' \
  -d '{
    "input_type": "blob",
    "container_name": "imaging-pipeline",
    "input_path": "Raw_Input/Run1/Batch1/DEID_PNGs",
    "output_path": "Processed/Run1",
    "skip_ocr": true,
    "only": ["ocr_quality", "page_subtype", "encounter_type", "page_sequencing"]
  }'
```

4. Review-ui → Codeable / Encounter Type / Actual Sequence populate from the
   new CSVs (or DB in production mode)
