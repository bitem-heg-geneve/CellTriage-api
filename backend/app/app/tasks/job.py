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
    job_update = JobUpdate(status="done", process_end_at=datetime.now())
    crud.job.update(db=db, db_obj=job, obj_in=job_update)
    logger.info(f"Process_job_end, job_id:{job.id}")

    db.close()
    return job_id
