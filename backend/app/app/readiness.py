"""Readiness gate for the inference worker (shared triage guard, §4).

Runs before the Celery infer worker starts (docker-compose.yml, worker-infer):

    python -m app.readiness && exec celery -A app.main.celery worker -Q infer ...

It exits non-zero unless ALL of the following hold:

1. Model: LOAD_CT_TAGGER=yes, the checkpoint file exists and the CtTagger loads.
2. Model behaviour: the two control texts in readiness_controls.json score as
   expected: positive > negative and positive >= CELLTRIAGE_READINESS_MIN
   (default 0.6). A wrong or degraded model fails here.
3. Fetch: SIBILS_URL returns a record with title or abstract for the positive
   control's PMID. The SIBiLS instance and collection versions are printed and
   recorded. A stale or unreachable text source (the 25 Oct 2026 incident: a
   stale instance, no text for recent PMIDs, 0.0 for every article) fails here.
   CELLTRIAGE_READINESS_FETCH=0 skips this step, loudly, for offline dev only.

The outcome is written best-effort to the `readiness` table (row "worker-infer")
so the API's GET /api/v1/readyz can report it; a database failure does not
change the exit code. With `restart: always` a failing gate shows up as a
restarting container, never as a worker that scores.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NoReturn, Optional

from app.core.config import settings

CONTROLS_PATH = Path(__file__).with_name("readiness_controls.json")
RECORD_ID = "worker-infer"


class ReadinessError(Exception):
    """One failed check; main() records it and exits 1."""


def fail(msg: str) -> NoReturn:
    raise ReadinessError(msg)


def load_controls() -> dict:
    with CONTROLS_PATH.open(encoding="utf-8") as fh:
        c = json.load(fh)
    assert c["positive"]["text"] and c["negative"]["text"] and int(c["positive"]["pmid"]) > 0
    return c


def check_model(controls: dict) -> dict:
    """Steps 1 and 2. Returns what was measured, or raises ReadinessError."""
    import pandas as pd
    from app.tasks.ct_score import CHECKPOINT_PATH, LM_MODEL_NAME, TEXT_COL, get_tagger

    if os.environ.get("LOAD_CT_TAGGER", "no") != "yes":
        fail("LOAD_CT_TAGGER is not 'yes': this process would never load the model")
    if not os.path.isfile(CHECKPOINT_PATH):
        fail(f"model checkpoint {CHECKPOINT_PATH} not found (is model_resources mounted?)")

    t0 = time.monotonic()
    try:
        tagger = get_tagger()
        df = pd.DataFrame({
            "pmid": [int(controls["positive"]["pmid"]), 0],
            TEXT_COL: [controls["positive"]["text"], controls["negative"]["text"]],
        })
        predictions, _ = tagger.predict(df)
        scores = [float(s) for s in predictions.numpy().flatten().tolist()]
    except Exception as exc:  # noqa: BLE001 - any load/inference error is a failure
        fail(f"inference raised {type(exc).__name__}: {exc}")
    if len(scores) != 2 or not all(math.isfinite(s) for s in scores):
        fail(f"model returned no usable scores for the two controls: {scores}")

    pos, neg = scores
    min_pos = float(settings.CELLTRIAGE_READINESS_MIN)
    print(
        f"model ok: {CHECKPOINT_PATH} ({LM_MODEL_NAME}) positive-control={pos:.3f} "
        f"negative-control={neg:.3f} ({time.monotonic() - t0:.1f}s)"
    )
    if pos < min_pos:
        fail(f"positive control scored {pos:.3f} < {min_pos}: wrong or degraded model")
    if pos <= neg:
        fail(f"positive control ({pos:.3f}) does not outrank the negative control ({neg:.3f})")
    return {"checkpoint": CHECKPOINT_PATH, "lm_model": LM_MODEL_NAME,
            "positive": round(pos, 4), "negative": round(neg, 4), "min_positive": min_pos}


def fetch_control(pmid: int, retries: Optional[int] = None) -> dict:
    """Step 3, shared with the API's /readyz (which passes a small `retries` so a
    health probe answers within SIBILS_TIMEOUT instead of the full backoff
    cycle). Raises on any failure."""
    from app.tasks.sibils import Sibils, sibils_handler

    client = sibils_handler if retries is None else Sibils(
        max_concurrency=1, sibils_url=settings.SIBILS_URL, retries=retries)
    res = client.fetch([pmid], col="medline")
    record = res.found.get(str(pmid))
    doc = (record or {}).get("document") or {}
    if not record or not (doc.get("title") or doc.get("abstract")):
        raise RuntimeError(
            f"SIBiLS {settings.SIBILS_URL} returned no text for control PMID {pmid} "
            f"(warning={res.warning!r}, collection={res.collection_version})"
        )
    return {"url": settings.SIBILS_URL, "control_pmid": pmid, "sibils_version": res.sibils_version,
            "collection_version": res.collection_version, "warning": res.warning}


def check_fetch(controls: dict) -> Optional[dict]:
    if not settings.CELLTRIAGE_READINESS_FETCH:
        print("WARNING: SIBiLS fetch check skipped (CELLTRIAGE_READINESS_FETCH=0); dev only", file=sys.stderr)
        return None
    pmid = int(controls["positive"]["pmid"])
    try:
        info = fetch_control(pmid)
    except Exception as exc:  # noqa: BLE001
        fail(f"SIBiLS fetch check failed: {exc}")
    print(f"sibils ok: {info['url']} version={info['sibils_version']} collection={info['collection_version']}")
    return info


def record(result: dict) -> None:
    """Best-effort: store the outcome for /readyz. Never changes the exit code."""
    from app.db.session import SessionLocal
    from app.models.readiness import Readiness

    try:
        db = SessionLocal()
        try:
            db.merge(Readiness(
                id=RECORD_ID, ok=result["ok"], checked_at=result["checked_at"], error=result["error"],
                detail=json.dumps({"model": result.get("model"), "sibils": result.get("sibils")}, default=str),
            ))
            db.commit()
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not record readiness in the database: {exc}", file=sys.stderr)


def readiness_record(db) -> Optional[dict]:
    """The worker's last outcome as a dict for /readyz, or None if it never ran."""
    from app.models.readiness import Readiness

    row = db.get(Readiness, RECORD_ID)
    if row is None:
        return None
    detail = json.loads(row.detail) if row.detail else {}
    return {"id": row.id, "ok": row.ok, "checked_at": row.checked_at.isoformat() if row.checked_at else None,
            "error": row.error, "model": detail.get("model"), "sibils": detail.get("sibils")}


def main() -> int:
    controls = load_controls()
    result: dict = {"ok": False, "checked_at": datetime.now(), "error": None}
    try:
        result["model"] = check_model(controls)
        result["sibils"] = check_fetch(controls)
        result["ok"] = True
    except ReadinessError as exc:
        result["error"] = str(exc)
        record(result)
        print(f"READINESS FAIL: {exc}", file=sys.stderr)
        return 1
    record(result)
    print("READINESS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
