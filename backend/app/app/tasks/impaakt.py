import logging
from celery import shared_task
from celery.utils.log import get_logger
from app import crud
from app.db.session import SessionLocal
from app.schemas.source import SourceUpdate
import pickle

MODEL_FP = r"/app/model_resources/impaakt.pckl"
model = pickle.load(open(MODEL_FP, "rb"))


@shared_task(
    name="ingress:job_impaakt",
    bind=True,
    default_retry_delay=30,
    max_retries=3,
    soft_time_limit=10000,
)
def job_impaakt(self, job_id):
    db = SessionLocal()
    job = crud.job.get(db=db, id=job_id)

    for source in job.sources:
        if source.text:
            probas = model.predict_proba([source.text])
            score = float(probas[0][1]) * 100
            source_update = SourceUpdate(impaakt_score=score)
            crud.source.update(db=db, db_obj=source, obj_in=source_update)

    db.close()
    return job_id
