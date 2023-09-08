import asyncio
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app import crud
from app.api import deps
from app.schemas.job import Job, JobCreate, JobStatus
from app.crud import job, article
from app.schemas.article import ArticleUpdate
from app.tasks.job import job_process_start, job_process_end
# from app.tasks.crawl import job_crawl
# from app.tasks.impaakt import job_impaakt
# from app.tasks.NER import job_ner
# from app.tasks.company import job_company
from app.tasks.text import job_text
from app.tasks.ct_score import job_ct_score
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
    Create a CellTriage job including a list of candicate articles. Each article must include a pmid. The articles will be scored for relevancy for Cellosaurus curation.\n
    
    By default the system will attempt to extract the fulltext from PubMed Central. If this is not possible then the text will be exracted from the article abstract only. If the job -option "use_fulltext" is set to false then the text will be retrieved from the abstract for all articles.\n 
     
    
    """
    job = crud.job.create(db=db, obj_in=job_in)

    # TODO
    chain(
        job_process_start.s(job.id),
        job_text.s(),
        job_ct_score.s(),
        job_process_end.si(job.id),
    ).apply_async() 

    # group_1 = []
    # if job.impaakt_ranking:
    #     group_1.append(job_impaakt.s())

    # chain_1_a = []
    # if job.named_entity_recognition:
    #     chain_1_a.append(job_ner.s())

    # if job.company_classification:
    #     chain_1_a.append(job_company.s())

    # if len(chain_1_a) > 0:
    #     group_1.append(chain(chain_1_a))
   
    # if len(group_1) > 0:
    #     chain(
    #         job_process_start.s(job.id),
    #         job_crawl.s(),
    #         group(group_1),
    #         job_process_end.si(job.id),
    #     ).apply_async()
    # else:
    #     # no tasks for Impaakt, NER, Company (or SASB) models 
    #     chain(
    #         job_process_start.s(job.id),
    #         job_crawl.s(),
    #         job_process_end.si(job.id),
    #     ).apply_async()

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
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Retrieve job by ID\n
    Optionally includes text
    """
    job = crud.job.get(db=db, id=job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")

    if not include_text:
        for article in job.article_set:
            article.text = None

    return job
