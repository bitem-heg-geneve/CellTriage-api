import logging
from celery import shared_task
from celery.utils.log import get_logger
from datetime import datetime
import httpx
from app.models.job import Job
from app.schemas.job import JobUpdate
# from app.schemas.source import SourceUpdate
from app.schemas.article import ArticleUpdate
from app import crud
from sqlalchemy.orm import Session
from app.api import deps
from app.db.session import SessionLocal
from app.services.job_summary import STATUS_FAILED, article_row, summarize
# from app.schemas import Source
# from app.schemas import Article

# from celery.result import AsyncResult
from celery import chord

logger = get_logger(__name__)


@shared_task(
    name="ingress:job_process_start",
    bind=True,
    default_retry_delay=30,
    max_retries=3,
    soft_time_limit=10000,
)
def job_process_start(self, job_id):
    db = SessionLocal()
    job = crud.job.get(db=db, id=job_id)
    job_update = JobUpdate(status="in progress", process_start_at=datetime.now())
    crud.job.update(db=db, db_obj=job, obj_in=job_update)
    logger.info(f"Process_job_start, job_id:{job.id}")

    db.close()
    return job_id


@shared_task(
    name="ingress:job_process_end",
    bind=True,
    default_retry_delay=30,
    max_retries=3,
    soft_time_limit=10000,
)
def job_process_end(self, job_id):
    db = SessionLocal()
    job = crud.job.get(db=db, id=job_id)
    # Shared triage guard §3: the final status says what the articles say:
    # done (all scored), partial (some not), failed (none, or an endpoint problem).
    summary = summarize((article_row(a) for a in job.article_set), final=True)
    job_update = JobUpdate(status=summary.status, error=summary.error, process_end_at=datetime.now())
    crud.job.update(db=db, db_obj=job, obj_in=job_update)
    logger.info(
        "Process_job_end, job_id:%s status=%s n_submitted=%d n_scored=%d n_no_text=%d "
        "n_fetch_error=%d n_no_prediction=%d error=%s",
        job.id, summary.status, summary.n_submitted, summary.n_scored, summary.n_no_text,
        summary.n_fetch_error, summary.n_no_prediction, summary.error,
    )
    if summary.status == STATUS_FAILED:
        logger.error("job %s failed: %s (%s)", job.id, summary.error, job.sibils_warning)

    db.close()
    return job_id
