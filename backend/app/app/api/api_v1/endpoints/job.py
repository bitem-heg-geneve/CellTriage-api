import asyncio
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app import crud
from app.api import deps
from app.schemas.job import Job, JobCreate, JobStatus
from app.crud import job, source
from app.schemas.source import SourceUpdate
from app.tasks.job import job_process_start, job_process_end
from app.tasks.crawl import job_crawl
from app.tasks.impaakt import job_impaakt
from app.tasks.NER import job_ner
from app.tasks.company import job_company
from typing import List, Union
from celery import chain, group

router = APIRouter()


@router.post("", status_code=200, response_model=JobStatus)
def create_job(
    *,
    job_in: JobCreate,
    db: Session = Depends(deps.get_db),
) -> dict:
    # ) -> Any:
    """\
    Create an Impaakt job including a list of candicate sources. 
    1. Each source must include an url \n
    2. For each source a text can optionally be included in the request. For sources for which no text is provided, the system will attempt to crawl the url and extract text from either html or PDF documents. \n
    3. Impaakt ranking is default but optional. Only sources with text will be processed.
    4. Named entity recognition (NER) is default but optional. For each source a NER-list can be included in the request. For sources for which no NER-list is provided the system will attempt to extract entities. \n Only sources with text will be processed. \n
    5. Company classification is default but optional. Only sources with a NER-list will be processed.
    
    """
    job = crud.job.create(db=db, obj_in=job_in)

    group_1 = []
    if job.impaakt_ranking:
        group_1.append(job_impaakt.s())

    chain_1_a = []
    if job.named_entity_recognition:
        chain_1_a.append(job_ner.s())

    if job.company_classification:
        chain_1_a.append(job_company.s())

    if len(chain_1_a) > 0:
        group_1.append(chain(chain_1_a))
   
    if len(group_1) > 0:
        chain(
            job_process_start.s(job.id),
            job_crawl.s(),
            group(group_1),
            job_process_end.si(job.id),
        ).apply_async()
    else:
        # no tasks for Impaakt, NER, Company (or SASB) models 
        chain(
            job_process_start.s(job.id),
            job_crawl.s(),
            job_process_end.si(job.id),
        ).apply_async()

    return job


@router.get("/{job_id}/status", status_code=200, response_model=JobStatus)
def job_status(
    *,
    job_id: int,
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Retrieve job status by job_id.
    """
    result = crud.job.get(db=db, id=job_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")

    return result


@router.get(
    "/{job_id}",
    status_code=200,
    response_model=Job,
    response_model_exclude_none=True,
)
def job_details(
    *,
    job_id: int,
    include_text: bool = False,
    include_NER: bool = False,
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Retrieve job by ID\n
    Optionally includes text and NER.
    """
    job = crud.job.get(db=db, id=job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")

    if not include_text:
        for source in job.sources:
            source.text = None

    if not include_NER:
        for source in job.sources:
            source.NER = []

    return job
