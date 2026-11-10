"""Liveness and readiness (shared triage guard, §4)."""
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.api import deps
from app.core.config import settings
from app.readiness import fetch_control, load_controls, readiness_record

router = APIRouter()


@router.get("/healthz", status_code=200)
def healthz() -> dict:
    """Liveness: the API process answers. See /readyz for the service."""
    return {"status": "ok"}


@router.get(
    "/readyz",
    summary="Readyz",
    description="Readiness: SIBILS_URL answers for the control PMID (its instance and collection "
                "versions are reported), and the infer worker's last `python -m app.readiness` "
                "(model + fetch check) passed. 503 with a `problems` list otherwise.",
)
def readyz(db: Session = Depends(deps.get_db)):
    problems = []
    sibils: dict = {"url": settings.SIBILS_URL}
    try:
        sibils = fetch_control(int(load_controls()["positive"]["pmid"]), retries=1)
    except Exception as exc:  # noqa: BLE001 - any failure is "not ready"
        problems.append(f"sibils: {exc}")

    worker = readiness_record(db)
    if worker is None:
        problems.append("worker: no readiness record yet (python -m app.readiness has not run)")
    elif not worker.get("ok"):
        problems.append(f"worker: {worker.get('error') or 'readiness failed'}")

    body = {"status": "ok" if not problems else "not_ready", "problems": problems,
            "sibils": sibils, "worker": worker}
    return JSONResponse(body, status_code=200 if not problems else 503)
