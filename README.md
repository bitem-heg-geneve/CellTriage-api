# CellTriageAPI

## Deployment
```
git clone https://github.com/bitem-heg-geneve/CellTriage-api.git

chmod 777 ./backend/app/run.sh

unzip ./backend/app/model_resources/impaakt.zip

docker compose -f "docker-compose.d`
```

### Upgrading a running stack

The `api` container runs `alembic upgrade head` on start (`prestart.sh`); the
workers do not. After pulling a version that adds a migration, start the API
first, then (re)start the workers:

```
git pull
docker compose build
docker compose up -d api              # runs the migrations
docker compose up -d worker-ingress worker-infer
docker compose logs -f worker-infer   # must print "READINESS OK" before the Celery banner
curl -i http://127.0.0.1:8001/api/v1/readyz
```

### Configuration (`compose/.env.dev`)

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | **none** (required) | Postgres URL |
| `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND` | `redis://127.0.0.1:6379/0` | Celery broker / backend |
| `MAX_INGRESS_CONCURRENCY` | **none** (required) | Parallel SIBiLS requests per ingress task |
| `SIBILS_URL` | **none** (required) | SIBiLS API base URL, e.g. `https://biodiversitypmc.sibils.org/api`. The process refuses to start without it or with a non-http(s) value; a trailing slash is stripped (the client appends `/fetch`) |
| `SIBILS_TIMEOUT` | `30` | Seconds, connect and read, per SIBiLS request |
| `SIBILS_RETRIES` | `5` | Attempts per SIBiLS batch (backoff 2 s → 30 s, about one minute in all) before its articles are `fetch_error` |
| `LOAD_CT_TAGGER` | `no` | `yes` on `worker-infer` only: load the model |
| `CELLTRIAGE_READINESS_MIN` | `0.6` | Minimum score of the positive control in `app.readiness` |
| `CELLTRIAGE_READINESS_FETCH` | `1` | `0` skips the SIBiLS part of `app.readiness` (offline development only) |

## API
This API allows to create literature triage jobs for the Cellosaurus knowledge resource on cell lines. Article relevancy scores are estimated using state-of-the art language models, fine-tuned for the triage of both abstracts and fulltext.

A score is only ever produced from text the service actually fetched from
SIBiLS. An article without text, or without a prediction, is reported as an
error, never as a number (`score: null` + `error`), and the job's `status`
and counts say how many were affected. This is the shared pre-scoring guard of
the triage services (INFRA spec, 2026-11-10), after the incident of 25 Oct 2026
where a stale SIBiLS instance returned no text for recent PMIDs and every
article was published with `score: 0.0`.

### Job Create endpoint
**URL:** POST `/api/v1/jobs`

Create a CellTriage job including a list of candicate articles. Each article must include a pmid. The articles will be scored for relevancy for Cellosaurus curation.

By default the system will attempt to extract the fulltext from PubMed Central. If this is not possible then the text will be exracted from the article abstract only. If the job -option "use_fulltext" is set false then the text will be retrieved from the abstract for all articles.

Example request body:
```
{
  "use_fulltext": true,
  "article_set": [
    {
      "pmid": 36585756
    },
    {
      "pmid": 36564873
    }
  ]
}
```

The response is the job status (see below) with `status: "pending"`.

### Job Details endpoint
**URL:** GET `/api/v1/jobs/{job_id}` (add `?include_text=true` for the text)

The article scores can be used to rank articles by estimated triage relavancy.
`text_source` denotes if the text was retrieved from the article abstract or from fulltext.

Per article:

| Field | Meaning |
|---|---|
| `score` | Relevancy in [0, 1], rounded to 2 decimals, or `null` when the article could not be scored |
| `text_status` | What the SIBiLS fetch gave: `ok` (title + abstract or full text), `missing` (SIBiLS did not return the PMID), `empty` (record without title and abstract), `fetch_error` (SIBiLS unreachable after retries) |
| `error` | Why `score` is `null`: `missing`, `empty`, `fetch_error`, or `no_prediction` (the article had text but the model gave no usable score) |

Per job:

| Field | Meaning |
|---|---|
| `status` | `pending` → `in progress` → `done` (every article scored), `partial` (some not; the scored ones are valid) or `failed` (none scored, or more than half of a job of 20 or more articles has no text: an endpoint or collection problem, not a few odd PMIDs) |
| `error` | `sibils_fetch` or `no_prediction` when the job failed |
| `n_submitted`, `n_scored`, `n_no_text` (missing + empty), `n_fetch_error`, `n_no_prediction` | Counts over the articles |
| `sibils_version`, `collection_version`, `sibils_warning` | What the SIBiLS instance answered for this job (`sibils_warning` holds its `missing ids:…` text, or the fetch error) |

A client should rank on `score` where it is not `null`, and skip or re-submit
the others; a `failed` job should not be published.

Example response (two PMIDs SIBiLS has, one it does not):
```
{
  "id": 42,
  "use_fulltext": false,
  "status": "partial",
  "error": null,
  "n_submitted": 3,
  "n_scored": 2,
  "n_no_text": 1,
  "n_fetch_error": 0,
  "n_no_prediction": 0,
  "sibils_version": "v5.1.5.10",
  "collection_version": "med_r1_v5.10",
  "sibils_warning": "missing ids:999999999",
  "job_created_at": "2026-11-10T14:15:19.327588",
  "process_start_at": "2026-11-10T14:15:19.351811",
  "process_end_at": "2026-11-10T14:16:02.379581",
  "process_time": 43.03,
  "article_set": [
    {
      "pmid": 42967740,
      "score": 0.12,
      "pmcid": "",
      "entrez_date": "2026-09-30",
      "error": null,
      "text_status": "ok",
      "text_source": "abstract"
    },
    {
      "pmid": 42989505,
      "score": 0.37,
      "pmcid": "PMC13740358",
      "entrez_date": "2026-11-05",
      "error": null,
      "text_status": "ok",
      "text_source": "abstract"
    },
    {
      "pmid": 999999999,
      "score": null,
      "pmcid": null,
      "entrez_date": null,
      "error": "missing",
      "text_status": "missing",
      "text_source": null
    }
  ]
}
```

### Job Status endpoint
**URL:** GET `/api/v1/jobs/{job_id}/status`

Same fields as above without `article_set`.

### Health
**URL:** GET `/api/v1/healthz` – liveness: the API process answers (`{"status": "ok"}`).

**URL:** GET `/api/v1/readyz` – readiness: `SIBILS_URL` answers with a record for
the control PMID (the live `sibils_version` and `collection_version` are
reported), and the infer worker's last readiness gate passed. 200 when ready,
503 with a `problems` list otherwise:
```
{
  "status": "ok",
  "problems": [],
  "sibils": {"url": "https://biodiversitypmc.sibils.org/api", "control_pmid": 37535222,
             "sibils_version": "v5.1.5.10", "collection_version": "med_r1_v5.10", "warning": null},
  "worker": {"id": "worker-infer", "ok": true, "checked_at": "2026-11-10T14:16:12", "error": null,
             "model": {"positive": 0.97, "negative": 0.03, "min_positive": 0.6, "...": "..."},
             "sibils": {"...": "..."}}
}
```

## Readiness gate (worker-infer)

`python -m app.readiness` runs before the Celery infer worker starts
(`docker-compose.yml`, `worker-infer.command`). It exits non-zero, and the
container restarts instead of serving, unless:

1. `LOAD_CT_TAGGER=yes`, the checkpoint exists and the model loads;
2. the two control texts in `backend/app/app/readiness_controls.json` score as
   expected: the positive one (a cell line establishment paper, PMID 37535222)
   ≥ `CELLTRIAGE_READINESS_MIN` (0.6) and above the negative one (an unrelated
   clinical trial abstract);
3. `SIBILS_URL` returns a record with text for the positive control's PMID; the
   SIBiLS instance and collection versions are printed and stored.

The outcome is stored in the `readiness` table for `/api/v1/readyz`. To run it
by hand: `docker compose exec worker-infer python -m app.readiness`.

## Tests

```
cd backend/app
python -m pytest tests/test_scoring_guard_units.py   # pure: job summary, SIBiLS response parsing, URL rule; no stack, no torch
python -m pytest tests/test_scoring_guard.py         # against a running stack (CELLTRIAGE_BASE, default http://localhost:8001); skipped when none answers
python -m pytest tests/test_api.py                   # smoke tests against the running stack
```

`tests/test_scoring_guard.py` submits two PMIDs SIBiLS has (42967740, 42989505)
and one it does not (999999999) and checks `score: null` + `error: "missing"`
for the missing one, `status: "partial"` with `n_scored: 2`, `n_no_text: 1`,
and `/api/v1/readyz` → 200.
