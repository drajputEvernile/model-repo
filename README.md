# Advantmed Imaging Pipeline

Turns scanned medical charts into structured, reviewable data.

Point it at a folder of chart page images — a local directory or an Azure Blob
container — and it corrects each page, reads the text, discards the pages that
carry no information, confirms the chart belongs to the member it is supposed to,
and extracts the dates of service. Results land as CSVs beside the pages and in a
Postgres database, and come with a web viewer for checking any page against the
text that was read from it.

---

## What it produces

For every chart, under its own folder name:

| Output | Contains |
|---|---|
| `ocr/…_prelim.txt`, `…_final1.json`, `…_final2.json` | the text read from the chart, at each of the three OCR passes (final1/final2 are JSON with `pages[].content`) |
| `imaging/…_rotation.csv` | orientation and mirroring corrected per page |
| `imaging/…_hw_printed.csv` | whether each page is printed, handwritten or mixed |
| `imaging/…_junk.csv` | pages judged blank, junk or duplicate, and why |
| `imaging/…_member_extraction.csv` | member names, dates of birth and ids found on each page |
| `imaging/…_member_verification.csv` | per-page match against the expected member |
| `imaging/…_dos.csv` | dates of service, with the keyword each was found near |
| `corrected-pages/` | the de-skewed, correctly oriented page images |

Each chart also gets an **accept / needs-review / reject** decision from member
verification, with the reason recorded rather than implied.

---

## How a chart is processed

Eight stages, in order. Every page carries its own status, so one unreadable page
is recorded and skipped rather than failing the chart.

| # | Stage | Runs on | Produces |
|---|---|---|---|
| 1 | Rotation + handwriting detection | every page | orientation correction, printed/handwritten label |
| 2 | Preliminary OCR | every page | first-pass text |
| 3 | Blank / junk / duplicate — pass 1 | printed pages | candidates to drop |
| 4 | Final OCR 1 | survivors + handwritten | second-pass text |
| 5 | Final OCR 2 | survivors + handwritten | third-pass text, for difficult pages |
| 6 | Blank / junk / duplicate — pass 2 | handwritten + survivors | final keep/drop verdict |
| 7 | Member extraction + verification | pages that survived | the accept/reject decision |
| 8 | Date of service | pages that survived | service dates per page and per document |

**Stage 5 is billed per page.** Re-running a chart therefore **resumes** by
default: pages already completed are not sent again. Reprocessing everything is an
explicit choice.

The member manifest — the roster a chart is checked against — loads independently,
before or after the pages. If it arrives late, re-run stage 7 alone.

---

## Requirements

| | |
|---|---|
| Database | PostgreSQL 14+ |
| Runtime | Docker, or Python 3.12 directly (3.13+ is not supported) |
| Optional | Azure Blob Storage for intake, Azure Document Intelligence for stage 5 |

Every optional service is genuinely optional. When one is absent the affected
stage degrades in a way the run records — the output says which path produced it,
so a partially configured environment is visible in the data rather than silent.
`GET /health` lists what is switched on and names the missing precondition for
anything that is not.

---

## Quick start

Two services, each with its own `docker-compose.yml`. They share a database and a
chart workspace, and never call each other.

- **`core-pipeline`** — intake and the eight stages. Port 8001.
- **`review-ui`** — read-only viewer over the results. Ports 3000 (API) and 3001 (web).

```bash
# 1. Create the schema — once, before either service starts
psql "$DATABASE_URL" -f schema/v1.sql

# 2. The pipeline
cd core-pipeline
cp .env.example .env          # set DATABASE_URL, plus any Azure credentials
docker compose up -d --build  # http://localhost:8001/docs

# 3. The viewer
cd ../review-ui
cp .env.example .env
docker compose up -d --build  # http://localhost:3001
```

On Windows PowerShell, use `$env:DATABASE_URL` and `Copy-Item .env.example .env`;
everything else is identical.

To run the services directly instead of in Docker — with reload, on any of the
three platforms — see
[Mode A — Local](docs/API.md#mode-a--local-macos--windows--linux).

---

## Running a chart

Two operations cover the service. Each is an HTTP endpoint and a CLI subcommand
with the same options. Write is part of both — pass a write path and missing
destination files are written, existing ones skipped.

| | |
|---|---|
| `POST /api/charts/run` | one chart in, through the eight stages, optionally written back out; resume with `chart_id` / `chart_name` |
| `POST /api/charts/batch-run` | the same, once per subfolder of a drop |

A source is a path plus a folder name, and **the folder name is the chart name** —
it identifies the chart in the database, prefixes every output file, and is the key
the member manifest is matched on.

```bash
# From a folder on the server
curl -X POST localhost:8001/api/charts/run -H 'Content-Type: application/json' \
  -d '{"local_read_path":"/data/inbox",
       "local_folder_name":"52743839_44976074",
       "local_write_path":"/data/outbox"}'

# From Azure Blob Storage
curl -X POST localhost:8001/api/charts/run -H 'Content-Type: application/json' \
  -d '{"blob_container":"imaging-pipeline",
       "blob_read_path":"run1/batch1",
       "blob_read_folder_name":"52743839_44976074",
       "blob_write_path":"Processed/Run1"}'
```

The call returns immediately; poll
`GET /api/charts/by-name/52743839_44976074` for progress.

Add `"through": "ocr_final2"` to stop after a stage, or `"only": ["dos_extract"]`
to run one stage on its own against results already on disk.

Full reference — every field, every status code, batches, partial runs and the CLI:
[`docs/API.md`](docs/API.md#how-to-run-a-chart).

---

## Reviewing the results

`review-ui` at <http://localhost:3001> lists every chart the pipeline has produced,
and shows each page beside the text read from it, the stage badges it earned, and
the member verification outcome. It mounts the chart workspace **read-only** and
records no decisions — it is for checking the pipeline's work, not for annotating
it.

It can read results either from the database or straight from the output files, so
it can be pointed at a finished drop without a database at all.

---

## Deploying this securely

Two things to settle before this handles real charts outside a trusted network.
Both are deliberate, current limitations rather than oversights:

- **Neither service authenticates requests, and neither terminates TLS.** Charts
  carry member names and dates of birth. Put both services behind a reverse proxy
  that provides authentication and TLS before exposing either port beyond the host.
  The viewer's login screen keeps a casual visitor off the page; it is not an
  access control, and the API behind it is reachable without it.
- **The key/value extraction is required.** One stage right after OCR finds the member
  name / DOB / ID, DOS, page numbers and headings that member verification, DOS and
  sequencing read. It needs its packages (`requirements-extraction.txt`) and weights
  (`models/gliner_low`, `layout_heron`, `kv_ranker/v002`); without them the chart fails at
  that stage. [`GET /health`](docs/API.md#get-health) → `extraction` reports whether it can run.
  Pages Azure never read (high-quality printed pages) have no word boxes and are not
  extracted — see [`docs/EXTRACTION.md`](docs/EXTRACTION.md#pages-with-no-word-boxes).

---

## Verifying an installation

```bash
python -m pytest tests/ -q
```

No database or cloud credentials required. `scripts/simulate_extraction.py` runs a prepared
document through the extraction and every stage that uses it (needs the model weights).

---

## Documentation

| Document | Covers |
|---|---|
| [`docs/HOW_TO_RUN.md`](docs/HOW_TO_RUN.md) | **Re-run recipes** — skip OCR, `only` / `force` / batch on charts that already finished OCR |
| [`docs/API.md`](docs/API.md) | Running both services, full API and CLI reference, troubleshooting |
| [`docs/EXTRACTION.md`](docs/EXTRACTION.md) | The key/value extraction stage, staging, models, training |
| [`docs/FLOW.md`](docs/FLOW.md) | What runs when — end-to-end diagrams, skip rules, resume behaviour |
| [`docs/LOGIC.md`](docs/LOGIC.md) | How each decision is made, and exactly what it writes |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System shape, data model, the role of every file |
| [`docs/SCALING.md`](docs/SCALING.md) | Design: running across several machines. Opens with a plain-language half for non-technical readers |
