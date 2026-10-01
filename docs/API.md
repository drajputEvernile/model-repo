# API & running

Two services, separate deploys — they never call each other. They share Postgres
and `data/folders`.

| Service | Local port | Swagger |
|---|---|---|
| core-pipeline | `8001` | http://localhost:8001/docs |
| review-ui API | `8002` | http://localhost:8002/docs |
| review-ui web | `5174` | http://localhost:5174 |

Algorithms: [LOGIC.md](LOGIC.md). Shape: [ARCHITECTURE.md](ARCHITECTURE.md).

**Windows tips (every command below):** use `curl.exe` (not `curl`),
`py -3.12` (not plain `python` if 3.13 is default),
`.venv\Scripts\Activate.ps1`, and `$env:NAME = "value"` for env vars.
If `Activate.ps1` is blocked:
`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

---

## Local: Environment set up

### 1. System tools

| Tool | macOS | Linux | Windows |
|---|---|---|---|
| **Python 3.12** (not 3.13+) | `brew install python@3.12` | `apt install python3.12 python3.12-venv` | `winget install Python.Python.3.12` |
| `tesseract` | `brew install tesseract` | `apt install tesseract-ocr` | [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) |
| `psql` | `brew install libpq` | `apt install postgresql-client` | PostgreSQL installer |
| Node 20+ (frontend) | `brew install node` | `apt install nodejs npm` | `winget install OpenJS.NodeJS.LTS` |
| Git LFS (HW `.pth`) | `brew install git-lfs` | `apt install git-lfs` | `winget install GitHub.GitLFS` |

Set `TESSERACT_CMD` when `tesseract` is not on `PATH` (always on Windows), e.g.
`TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`.

### 2. Database schema

`schema/` has: `clear_schema.sql`, `v1.sql`, `v2.sql`, `patch_output_path.sql`.

**First-time setup** (empty database) — apply V1; V2 is optional:

```bash
# macOS / Linux
psql "$DATABASE_URL" -f schema/v1.sql            # required — all implemented tables
psql "$DATABASE_URL" -f schema/v2.sql            # optional — proposals only
psql "$DATABASE_URL" -c "SELECT stage_name, pass_no, seq, is_phase1 FROM pipeline_stage ORDER BY seq;"
# Expect 12 phase-1 rows from v1; + rejection_logic if v2 was applied.
```

```powershell
# Windows
psql $env:DATABASE_URL -f schema/v1.sql
psql $env:DATABASE_URL -f schema/v2.sql
psql $env:DATABASE_URL -c "SELECT stage_name, pass_no, seq, is_phase1 FROM pipeline_stage ORDER BY seq;"
```

**Existing database** (already applied an older v1 / early v2) — do **not** re-run
`v1.sql` (CREATE TABLE will fail). Apply the additive patch instead:

```bash
psql "$DATABASE_URL" -f schema/patch_output_path.sql
```

That patch is idempotent and:

| Change | Detail |
|---|---|
| `chart_list.output_path` | Column if missing; write path stored on ingest |
| `chart_list.status` | Legacy `'rejected'` → `'completed'` |
| `pipeline_stage` | Registers / promotes `page_subtype`, `encounter_type`, `page_sequencing` (phase-1) |
| **New tables** | `encounter_type_results`, `page_sequencing_results` (moved out of older v2 proposals) |
| `page_stage_status` | Seeds `pending` rows for those three stages on existing pages |

When a write path is passed on run/batch-run it is stored **as-is** (chart folder
appended if missing). Otherwise ingest derives e.g.
`Raw_Input/Run1/Batch1/DEID_Images/<chart>` → `Processed/Run1/Batch1/<chart>`.

Wipe and re-apply (full — drops roster too):

```bash
# macOS / Linux
psql "$DATABASE_URL" -f schema/clear_schema.sql
psql "$DATABASE_URL" -f schema/v1.sql
psql "$DATABASE_URL" -f schema/v2.sql
# or: ./scripts/reset_db.sh --yes
```

```powershell
# Windows
psql $env:DATABASE_URL -f schema/clear_schema.sql
psql $env:DATABASE_URL -f schema/v1.sql
psql $env:DATABASE_URL -f schema/v2.sql
# or: .\scripts\reset_db.ps1 -Yes   (if present)
```

Empty chart/OCR/imaging rows but **keep the manifest roster** (and
`pipeline_stage`):

```bash
psql "$DATABASE_URL" -f schema/clear_results_keep_manifest.sql
```

```powershell
psql $env:DATABASE_URL -f schema/clear_results_keep_manifest.sql
```

Empty `pipeline_stage` ⇒ `/ready` returns 503.

### 3. Python packages + `.env`

```bash
# macOS / Linux
cd core-pipeline
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set DATABASE_URL at minimum

cd ../review-ui/backend
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env    # or review-ui/.env.example
```

```powershell
# Windows
cd core-pipeline
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # set DATABASE_URL at minimum

cd ..\review-ui\backend
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item ..\.env.example ..\.env
```

Optional pip extras (local venv). **Docker always installs
`requirements-docling.txt` and `requirements-extraction.txt`.**

| File | Enables |
|---|---|
| `requirements-docling.txt` | Docling + ConvNeXt HW **packages** (weights separate) |
| `requirements-extraction.txt` | Key/value extraction (GLiNER, Heron, LightGBM) — **required**, see [EXTRACTION.md](EXTRACTION.md) |

Key `.env` knobs (paths relative to `core-pipeline/`):

| Variable | Default / notes |
|---|---|
| `DATABASE_URL` | required |
| `DATA_ROOT` | `../review-ui/data/folders` |
| `STAGE_WORKERS` / `BATCH_WORKERS` | `4` / `4` — keep `workers × STAGE_WORKERS ≤ DB_POOL_MAX`. For Docling heap corruption, try `BATCH_WORKERS=1` |
| `PYTHONFAULTHANDLER` / `OMP_NUM_THREADS` | `1` / `1` — abort dumps a Python traceback to stderr (`docker compose logs`); OpenMP stays single-threaded |
| `SKIP_OCR` | `false` — reuse on-disk `ocr/`; with `force:false` also gate-delta |
| `HW_MODEL_PATH` | `models/hw/handwritten_printed_convnext_tiny.pth` (falls back to `…_tiny_backup.pth`) |
| `RAPID_MODELS_DIR` | `models/rapidocr` |
| `SECTION_HEADER_MINILM_PATH` | `models/semantic-model` — local MiniLM (preferred) |
| `BLANK_JUNK_MODEL_DIR` | `models/blank-junk` — `tfidf_flat.joblib` + `default.json`. Missing file ⇒ regex rules only |
| `EXTRACTION_MODELS_ROOT` | `models` — holds `gliner_low/`, `layout_heron/`, `kv_ranker/v002/`. The version is fixed at v002 in `config.py`, not an env setting |
| `MODELS_HOST_PATH` | Docker only. Host folder mounted at `/app/core-pipeline/models`. Default `./models` |
| `SECTION_HEADER_SEMANTIC_ENABLED` | `true` — filter Final1 `section_headers` ≥90% |
| `DOCLING_TABLE_CELL_MATCHING` | `true` (default) — fill dense form table cells. Slower per page; Final1 falls back to RapidOCR-onnx only on a page timeout or a crash, never on short output. Set `false` for speed. Unrelated to the section-header RLock deadlock. |
| `DOCLING_IMAGES_SCALE` | `1.0` — keep at 1 for page images (`2` halves overlay boxes) |

Azure Blob / DocIntel are optional — missing ones degrade a stage in a
way the run records (`GET /health` names the gap). The key/value extraction is not
optional: without its packages or weights the `kv_extract` stage fails the chart.

Blob auth modes (`AZURE_STORAGE_AUTH`):

| Mode | When |
|---|---|
| `entra` (default) | `DefaultAzureCredential` (MI → CLI → …). Set `AZURE_CLIENT_ID` for a **user-assigned** MI |
| `managed_identity` | VM / App Service MI only — no CLI, no browser. Same `AZURE_CLIENT_ID` |
| `entra_interactive` | MI → `az login` → browser (dev machines) |
| `key` | `AZURE_STORAGE_ACCOUNT_KEY` |

`AZURE_PRINCIPAL_ID` is the object id used when assigning **Storage Blob Data
Reader/Contributor**; it is not passed to the SDK. Final2 DI features default
to `languages,barcodes` (`AZURE_DI_FEATURES=off` to disable).

---

## Prerequisites (Models)

Weight files are **not** on PyPI. Copy them into `core-pipeline/models/` on
Windows and on the Linux VM. Every model path in `.env` is **relative to
`core-pipeline/`** — the same values on both machines:

```
HW_MODEL_PATH=models/hw/handwritten_printed_convnext_tiny.pth
RAPID_MODELS_DIR=models/rapidocr
SECTION_HEADER_MINILM_PATH=models/semantic-model
BLANK_JUNK_MODEL_DIR=models/blank-junk
EXTRACTION_MODELS_ROOT=models
```

`blank-junk/` (`tfidf_flat.joblib` + `default.json`) is in git. The other
weight folders are not — copy those into `core-pipeline/models/` yourself.

Teammate preprocessing drop / HW+quality refresh checklist:
[`IMAGE_PREPROCESSING.md`](IMAGE_PREPROCESSING.md).

```
models/hw/handwritten_printed_convnext_tiny.pth         # preferred ConvNeXt (2026-09 drop)
models/hw/handwritten_printed_convnext_tiny_backup.pth  # prior ConvNeXt fallback
models/hw/metadata.json                                 # thresholds next to the .pth
models/hw/image_type_classification.pkl                 # RF fallback
models/rapidocr/
  PP-OCRv6_det_small.pth
  PP-OCRv6_rec_small.pth
  ch_ptocr_mobile_v2.0_cls_mobile.pth
  ppocrv6_dict.txt
models/gliner_low/        # GLiNER (key/value extraction)
models/layout_heron/      # Heron heading detector
models/kv_ranker/v002/    # trained extraction ranker (in git)
models/semantic-model/    # MiniLM — section_header_match --download
models/blank-junk/        # tfidf_flat.joblib + default.json (in git; path is BLANK_JUNK_MODEL_DIR)
```

**Handwritten / printed (ConvNeXt)** — copy the preferred `.pth` (+ `metadata.json`) into
`models/hw/`, then install torch:

```bash
# macOS / Linux
cd core-pipeline && source .venv/bin/activate
pip install -r requirements-docling.txt   # torch + torchvision + MiniLM
```

```powershell
# Windows
cd core-pipeline; .venv\Scripts\Activate.ps1
pip install -r requirements-docling.txt
```

Stage 1 also applies the preprocessing rule **Handwritten + High → Medium** on
`quality_tag` (score unchanged). After dropping new HW weights, refresh with
`"skip_ocr": true` (see [`HOW_TO_RUN.md` §4B](HOW_TO_RUN.md)).

| Missing | Fallback |
|---|---|
| ConvNeXt `.pth` / torch | RandomForest `models/hw/image_type_classification.pkl` |
| RapidOCR four files | **rapidocr-onnxruntime** (base `requirements.txt`) |
| GLiNER / Heron / `kv_ranker` | the `kv_extract` stage fails the chart, naming the missing folder |
| MiniLM / sentence-transformers | lexical header match (same 0.90 threshold) |
| `BLANK_JUNK_MODEL_DIR` / sklearn | regex blank/junk rules; reason starts with `regex_fallback:` |

### Section-header MiniLM

OCR (Final1/Final2) still pre-filters header candidates against
`stages/lib/keyword-canon/section_header_canon.json`, but the `section_headers` of the
OCR JSON are now written by the `kv_extract` stage (Heron detector + trained model), which
replaces them. Re-run only that stage — no re-OCR:

```bash
# API
curl -X POST localhost:8001/api/charts/run -H 'Content-Type: application/json' \
  -d '{"chart_id": 123, "only": ["kv_extract"]}'

# CLI
python cli.py run --chart-id 123 --only kv_extract
```

**Existing databases** (schema already applied): rename the stage once with
`psql "$DATABASE_URL" -f schema/patch_kv_extract_stage.sql` (same position, `kv_extract`;
safe to run twice).

**Recommended: keep weights under `models/semantic-model/`** (gitignored):

```bash
# macOS / Linux
cd core-pipeline && source .venv/bin/activate
pip install -r requirements-docling.txt          # sentence-transformers + hub
python -m stages.lib.ocr.section_header_match --download
python -m stages.lib.ocr.section_header_match --check
```

```powershell
cd core-pipeline; .venv\Scripts\Activate.ps1
pip install -r requirements-docling.txt
python -m stages.lib.ocr.section_header_match --download
python -m stages.lib.ocr.section_header_match --check
```

Equivalent Hub CLI (same destination):

```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir models/semantic-model
```

Runtime load order: `SECTION_HEADER_MINILM_PATH` (default `models/semantic-model`)
if present → else Hub id `SECTION_HEADER_MINILM_MODEL`. Disable with
`SECTION_HEADER_SEMANTIC_ENABLED=false`.

### RapidOCR download (ModelScope v3.9.2)

**macOS / Linux**

```bash
cd core-pipeline && mkdir -p models/rapidocr
BASE=https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2
curl -L -o models/rapidocr/PP-OCRv6_det_small.pth "$BASE/torch/PP-OCRv6/det/PP-OCRv6_det_small.pth"
curl -L -o models/rapidocr/PP-OCRv6_rec_small.pth "$BASE/torch/PP-OCRv6/rec/PP-OCRv6_rec_small.pth"
curl -L -o models/rapidocr/ch_ptocr_mobile_v2.0_cls_mobile.pth \
  "$BASE/torch/PP-OCRv4/cls/ch_ptocr_mobile_v2.0_cls_mobile.pth"
curl -L -o models/rapidocr/ppocrv6_dict.txt \
  "$BASE/paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt"
pip install -r requirements-docling.txt
```

**Windows (PowerShell)** — `BASE=...` is bash-only; use `$base` and `curl.exe`:

```powershell
cd core-pipeline
New-Item -ItemType Directory -Force -Path models\rapidocr | Out-Null
$base = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2"
curl.exe -L -o models\rapidocr\PP-OCRv6_det_small.pth "$base/torch/PP-OCRv6/det/PP-OCRv6_det_small.pth"
curl.exe -L -o models\rapidocr\PP-OCRv6_rec_small.pth "$base/torch/PP-OCRv6/rec/PP-OCRv6_rec_small.pth"
curl.exe -L -o models\rapidocr\ch_ptocr_mobile_v2.0_cls_mobile.pth "$base/torch/PP-OCRv4/cls/ch_ptocr_mobile_v2.0_cls_mobile.pth"
curl.exe -L -o models\rapidocr\ppocrv6_dict.txt "$base/paddle/PP-OCRv6/rec/PP-OCRv6_rec_small/ppocrv6_dict.txt"
pip install -r requirements-docling.txt
```

### Key/value extraction (GLiNER, Heron, trained ranker)

See [EXTRACTION.md](EXTRACTION.md). Packages: `pip install -r requirements-extraction.txt`.
Weights go under `models/` (`gliner_low/`, `layout_heron/`, `kv_ranker/v002/`); the two public
models can be fetched at their pinned revisions with
`python -m stages.lib.extraction.util.model_setup`.
Confirm:

```bash
# macOS / Linux
curl -s localhost:8001/health | python -m json.tool
```

```powershell
# Windows
curl.exe -s localhost:8001/health | python -m json.tool
```

Check `docling_final1.ready`, `blank_junk_model.ready`, and `extraction.ready`.
`blank_junk_model.path` is the `tfidf_flat.joblib` under `BLANK_JUNK_MODEL_DIR`.

---

## Uvicorn / server start

### core-pipeline (`:8001`)

```bash
# macOS / Linux
cd core-pipeline && source .venv/bin/activate
python cli.py serve
# equivalent: uvicorn api.main:app --host 0.0.0.0 --port 8001
```

```powershell
# Windows
cd core-pipeline
.venv\Scripts\Activate.ps1
python cli.py serve
```

### review-ui

```bash
# macOS / Linux
# API — :8002 locally (:3000 in Docker)
cd review-ui/backend && source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8002 --reload

# Web — :5174 locally (:3001 in Docker)
cd review-ui/frontend && npm install && npm run dev
```

```powershell
# Windows
cd review-ui\backend
.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8002 --reload

# separate terminal
cd review-ui\frontend
npm install
npm run dev
```

Checks:

```bash
# macOS / Linux
curl -fsS localhost:8001/health
curl -fsS localhost:8001/ready    # 503 until DB + pipeline_stage are good
```

```powershell
# Windows
curl.exe -fsS localhost:8001/health
curl.exe -fsS localhost:8001/ready
```

`DATA_MODE=local` (review-ui default) reads `data/folders`.
`DATA_MODE=production` reads Postgres for OCR/imaging and **page images from
Azure Blob** using each chart’s `blob_container` + `blob_path` (set
`BLOB_ACCOUNT_URL` or `AZURE_STORAGE_ACCOUNT_NAME`, plus Entra credentials /
Managed Identity). Local `DATA_ROOT` is only a fallback when a workspace copy
exists.

POC login (not a security control): `imaging-user` / `aipocpw2026` — override
with `VITE_LOGIN_*` and rebuild the frontend.

---

## APIs available

Mutating chart calls return **202** and run in the background. Poll
`GET /api/charts/{id}` or `GET /api/charts/by-name/{name}`.

Stage names: `ocr_quality`, `ocr_prelim`, `blank_junk`, `ocr_final1`,
`ocr_final2`, `kv_extract`, `member_verify`, `dos_extract`, `page_subtype`,
`encounter_type`, `page_sequencing`. Pass 2: `blank_junk:2`.
Unknown name → **400**.

### `POST /api/charts/run` and `POST /api/charts/batch-run`

Both take the same small body. `run` does one chart (`chart_name`); `batch-run`
does every chart folder under `input_path` (or just `chart_list`). Any other
field is rejected with **422**, so an old body is reported, not half-applied.

| Field | run | batch-run | Default | What it does |
|---|---|---|---|---|
| `input_type` | required | required | — | `"local"` or `"blob"` |
| `container_name` | blob only | blob only | — | Azure container, used for read and write. Omit for local. |
| `input_path` | required | required | — | Folder holding the chart folder(s): a directory on the server (local — `C:/data/inbox` or `C:\\data\\inbox`) or a prefix in the container (blob) |
| `output_path` | optional | optional | no write | Results go to `<output_path>/<chart_name>/`, **replacing** existing files. Original pages are not re-sent. |
| `chart_name` | **required** | — | — | The chart folder under `input_path`. It is the chart's name everywhere. |
| `chart_list` | — | optional | all folders | Only these chart folders (list, or `"a,b"`). Names not found → `charts_missing`. |
| `sample` | — | optional | all | At most N charts, unfinished ones first |
| `only` | optional | optional | whole chain | Run just these stages, against what is already on disk |
| `run_through` | optional | optional | end of chain | Run from the top and stop after this stage |
| `skip_ocr` | optional | optional | `false` | Reuse existing OCR (workspace → Processed output → DB) instead of re-running the OCR engines — **no Azure final2 bill**. Quality/rotation and every non-OCR stage still re-run. |
| `skip_completed` | optional | optional | `false` | Skip charts whose status is `completed`, `needs_review` or `rejected`. run → **200** `{"status":"skipped"}`; batch → listed in `charts_skipped_completed`. When `false`, a batch re-runs those finished charts **first**, then the rest (each group alphabetical). |
| `skip_page_download` | optional | optional | **`true`** | Reuse page images already in the workspace. `false` = wipe `pages/` + `corrected-pages/` and fetch again from `input_path`. |

**What a full re-run clears** (no `only` / `run_through`) before the chain starts:

| Request | Kept | Cleared (DB rows + workspace files) |
|---|---|---|
| `skip_ocr: true` | `pages/`, `ocr/`, `ocr_results`, OCR stage status | every other result table, `imaging/`, `corrected-pages/` |
| default | `pages/` | every result table incl. `ocr_results`, `ocr/`, `imaging/`, `corrected-pages/` |
| `skip_page_download: false` | — | the above, plus `pages/` (re-fetched) |

`chart_list`, `page_list`, `manifest_member_list`, `page_ground_truth` and
`pipeline_jobs` are never cleared. `only` / `run_through` runs clear nothing —
they read what earlier stages left.

Not settable any more — fixed behaviour:

| Was | Now |
|---|---|
| `force` | Always reprocess every stage. `skip_ocr` is the only way to avoid re-OCR / re-billing. |
| `overwrite`, `write_mode` | Outputs always replace what is at the destination; original pages are never re-sent (`skip_orig_pages`). |
| `run_id`, `batch_id` | Always inferred from the path (`Run1`→`R1`, `Batch1`→`B1`). |
| `workers` | `BATCH_WORKERS` in `core-pipeline/.env`. Must satisfy `BATCH_WORKERS × STAGE_WORKERS + 2 ≤ DB_POOL_MAX`, else batch-run returns **400** naming the numbers. |
| `chart_id` resume | Run the same `input_path` + `chart_name` again: workspace pages are reused and every stage reprocesses. |
| `test_mode`, `skip_db_write` | Not on the API. |

Charts run in alphabetical order. Charts with more than `LARGE_CHART_MIN_PAGES`
(default 500) pages run strictly one at a time. Progress: `progress.txt` in the batch
parent folder and each chart's `imaging/progress.txt`.

```bash
# One local chart, written out
curl -X POST localhost:8001/api/charts/run -H 'Content-Type: application/json' \
  -d '{"input_type":"local","input_path":"/data/inbox",
       "chart_name":"52743839_44976074","output_path":"/data/processed"}'

# One blob chart, stop before the billed Azure OCR
curl -X POST localhost:8001/api/charts/run -H 'Content-Type: application/json' \
  -d '{"input_type":"blob","container_name":"imaging-pipeline",
       "input_path":"Raw_Input/Run1/Batch1/DEID_PNGs",
       "chart_name":"52754737_48221214","run_through":"ocr_final1"}'

# Batch: 3 charts, skip ones already finished, write results
curl -X POST localhost:8001/api/charts/batch-run -H 'Content-Type: application/json' \
  -d '{"input_type":"blob","container_name":"imaging-pipeline",
       "input_path":"Raw_Input/Run1/Batch1/DEID_PNGs",
       "output_path":"Processed/Run1/Batch1","sample":3,"skip_completed":true}'

# Batch: re-run only the classifiers on two charts, reusing OCR (no final2 bill)
curl -X POST localhost:8001/api/charts/batch-run -H 'Content-Type: application/json' \
  -d '{"input_type":"local","input_path":"/data/inbox",
       "chart_list":["52743839_44976074","52754737_48221214"],
       "skip_ocr":true,"only":["page_subtype","encounter_type","page_sequencing"]}'
```

```powershell
# Windows — one local chart, written out
curl.exe -X POST localhost:8001/api/charts/run -H "Content-Type: application/json" `
  -d "{\"input_type\":\"local\",\"input_path\":\"C:/data/inbox\",\"chart_name\":\"52743839_44976074\",\"output_path\":\"C:/data/processed\"}"

# Windows — every chart under a folder, skip finished ones
curl.exe -X POST localhost:8001/api/charts/batch-run -H "Content-Type: application/json" `
  -d "{\"input_type\":\"local\",\"input_path\":\"C:/data/inbox\",\"output_path\":\"C:/data/processed\",\"skip_completed\":true}"
```

`202` responses echo the options and, for batch, `charts_found`,
`charts_queued`, `charts_missing` and `charts_skipped_completed` (local drops
are counted up front; blob drops are listed in the background).

`/api/charts/batch` is a deprecated alias of `/batch-run`.

### Removed endpoints

| Path | Status | Use instead |
|---|---|---|
| `POST /api/charts/write` | **410** | `output_path` on `/run` or `/batch-run` |
| `POST /api/charts/{id}/rerun` | **410** | `POST /api/charts/run` with the same `input_path` + `chart_name` |

### `POST /api/manifest/sweep`

| Field | Required? | Default |
|---|---|---|
| `local_path` **or** `blob_container`+`blob_prefix` | one source | — |
| `run_id` / `batch_id` | optional | parsed from filename (`metadata_R1_B1.csv`) |
| `mirror_local` | optional | — | copy blob manifests into `METADATA_ROOT` |

### `POST /api/ground-truth/load`

Loads the client page spreadsheet into `page_ground_truth`. Synchronous. Apply
`schema/v1.sql` first on a database that predates the table.

| Field | Required? | Default |
|---|---|---|
| `local_path` | yes | CSV/XLSX file, or a directory of them |

`Chart_Name` is the chart folder. `Id` `1` matches `1.jpg`, `1.png`, and `1.tif`.
Member Name is Yes/No: Yes matches when the pipeline extracted a name, No when it did not.
Member DOB is compared as a date (`07/17/2025` matches `2025-07-17`); a shared year, month, or day is partial.
Re-running updates the same chart and page and leaves pages
that are not in the file.

```bash
python cli.py ground-truth --local /path/to/ground_truth.xlsx
```

### Read / ops

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | Liveness + optional feature readiness |
| `GET` | `/ready` | 503 unless DB up and `pipeline_stage` seeded |
| `GET` | `/api/stages` | Stage chain from DB |
| `GET` | `/api/charts/{id}` | Progress / status |
| `GET` | `/api/charts/by-name/{name}` | Same by chart name |
| `GET` | `/api/manifest/{record_id}` | Manifest rows: Postgres first, else scan `METADATA_ROOT` |
| `GET` | `/api/jobs?chart_id=` | Job log (`chart_id` optional) |

### review-ui (all `GET`, read-only)

| Path | Returns |
|---|---|
| `/api/health` | status + `data_mode` |
| `/api/folders` | charts + badges |
| `/api/folders/{id}` | pages + OCR availability |
| `/api/folders/{id}/pages/{n}/image` | page image |
| `/api/folders/{id}/ocr?kind=preliminary\|final1\|final2` | OCR text |
| `/api/folders/{id}/imaging` | imaging results |
| `/imaging/export.csv` | CSV export |

### CLI

The CLI keeps its own flags (`--resume`, `--through`, `--local-read-path`, …); it was not reshaped with the API body above.

```bash
# macOS / Linux
cd core-pipeline
python cli.py serve
python cli.py stages
python cli.py run --local-read-path /data/inbox --folder-name 52743839_44976074
python cli.py batch-run --local-read-path /data/inbox --sample 1 --through ocr_prelim
python cli.py write 52743839_44976074 --local-write-path /data/outbox
python cli.py rerun 7 --only dos_extract
python cli.py manifest --local ../review-ui/data/metadata/metadata_R1_B1.csv
```

```powershell
# Windows
cd core-pipeline
python cli.py serve
python cli.py stages
python cli.py run --local-read-path C:\data\inbox --folder-name 52743839_44976074
python cli.py batch-run --local-read-path C:\data\inbox --sample 1 --through ocr_prelim
python cli.py write 52743839_44976074 --local-write-path C:\data\outbox
python cli.py rerun 7 --only dos_extract
python cli.py manifest --local ..\review-ui\data\metadata\metadata_R1_B1.csv
```

### Utilities — load a folder of `metadata_Rn_Bn` files

```bash
# macOS / Linux
cd core-pipeline && source .venv/bin/activate
python ../utilities/load_metadata_manifests.py
python ../utilities/load_metadata_manifests.py /path/to/metadata/
```

```powershell
# Windows
cd core-pipeline
.venv\Scripts\Activate.ps1
python ..\utilities\load_metadata_manifests.py
python ..\utilities\load_metadata_manifests.py C:\path\to\metadata\
```

See [`utilities/README.md`](../utilities/README.md).

---

## Docker

Each service has its own `docker-compose.yml`. Order does not matter.

```bash
# macOS / Linux
cd core-pipeline && cp .env.example .env && docker compose up -d --build
curl -fsS localhost:8001/ready

cd ../review-ui && cp .env.example .env && docker compose up -d --build
# UI: http://localhost:4001   API: http://localhost:4000
# Compose pins DATA_MODE=local (disk). Production Mode is code-ready separately.
```

```powershell
# Windows
cd core-pipeline
Copy-Item .env.example .env
docker compose up -d --build
curl.exe -fsS localhost:8001/ready

cd ..\review-ui
Copy-Item .env.example .env
docker compose up -d --build
# UI: http://localhost:4001   API: http://localhost:4000
```

Local paths in API bodies must be paths **inside the container** (mount the
host folder first).

**Models:** copy the weight folders into `core-pipeline/models/` on the host.
`.env` paths stay relative (`models/blank-junk`, `models/hw/...`). Compose
mounts `MODELS_HOST_PATH` (default `./models`) at `/app/core-pipeline/models`.
The image does not contain the weights. Docling packages are in the image
already — rebuild once after pull:

```bash
docker compose up -d --build
curl -s localhost:8001/health | python -m json.tool   # docling_final1.ready
```

The key/value extraction packages are always in the image; its weights come from the
`models/` mount (`gliner_low/`, `layout_heron/`, `kv_ranker/v002/`).

---

## Tests

No database. Use Python 3.12 (the blank/junk tests load the joblib, which
needs scikit-learn 1.9.x). From the repo root:

```bash
python3.12 -m venv .venv-test && source .venv-test/bin/activate
pip install -r tests/requirements.txt
python -m pytest tests/ -q
```

```powershell
py -3.12 -m venv .venv-test
.venv-test\Scripts\Activate.ps1
pip install -r tests/requirements.txt
python -m pytest tests/ -q
```

Run this on the Windows machine after the model folders are in place. The
suite imports the pipeline modules; it does not call Azure or Postgres.
`tests/test_blank_junk_and_dos.py` loads `BLANK_JUNK_MODEL_DIR` when that
variable is set, otherwise `core-pipeline/models/blank-junk/`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/ready` → 503, empty `pipeline_stage` | `psql … -f schema/v1.sql` |
| Chart stuck at `ocr_prelim` | Install tesseract / set `TESSERACT_CMD` |
| `final2` empty | Set Azure DI endpoint + key |
| `manifest_missing` | Sweep manifest, then `rerun` with `only: ["member_verify"]` |
| `kv_extract` fails | `GET /health` → `extraction.reason` names the missing package or folder |
| Python 3.13 pip failure | Use 3.12 (`py -3.12` on Windows) |
| PowerShell `curl` oddities | Use `curl.exe` |
| `BASE is not recognized` | Bash-only; use `$base = "..."` in PowerShell |
| `Activate.ps1` blocked | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| HW `.pth` is ~1 KB after clone | `git lfs install` then `git lfs pull` |

**Pipeline file logs** (Docker): `core-pipeline/logs/core-pipeline.log`, rotated
at midnight to `core-pipeline.log.YYYY-MM-DD` (keep `LOG_BACKUP_DAYS`, default
30). Host path is `LOGS_HOST_PATH` (default `./logs`). Still also on stdout:
`docker compose logs -f api`. Log lines include a worker tag (`[batch-2]`,
`[page-0]`); set `LOG_COLOR=true` (Docker default) to colour those tags in the
terminal.
