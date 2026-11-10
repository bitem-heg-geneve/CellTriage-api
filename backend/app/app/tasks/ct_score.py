import logging
import math
from celery import shared_task
from celery.utils.log import get_task_logger
from app import crud
from app.db.session import SessionLocal
from app.schemas.article import ArticleUpdate
from app.services.job_summary import NO_PREDICTION, TEXT_OK, legacy_text_status
import pickle
import torch
from .ct_model import CtTagger
from transformers import BertForSequenceClassification, BertTokenizer
from app.tasks.BERT_1_1 import convert_examples_to_inputs, get_data_loader
import pandas as pd
import os

# Maximize CPU performance
torch.set_num_threads(16)  # Use all cores
torch.set_num_interop_threads(8)  # Parallel operations
# Enable optimizations
import torch._dynamo
torch._dynamo.config.suppress_errors = True

LABELS = ["pmid"]  # Dummy labels, not to be used
MAX_TOKEN_COUNT = 512
TEXT_COL = "text"
CHECKPOINT_PATH = "/model_resources/train_ml_ab_fulltext_pmbert/best-checkpoint.ckpt"
LM_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

tagger = None

log = get_task_logger(__name__)

def get_tagger():
    """Lazy-load CtTagger to avoid blocking worker startup."""
    global tagger
    if tagger is None and os.getenv("LOAD_CT_TAGGER", "no") == "yes":
        tagger = CtTagger(CHECKPOINT_PATH, LABELS, TEXT_COL, LM_MODEL_NAME)
    return tagger


def _usable_score(value) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and 0.0 <= f <= 1.0


def _mark_no_prediction(db, article):
    """§2: the article had text but the model gave no usable score: null + error."""
    crud.article.update(db=db, db_obj=article, obj_in=ArticleUpdate(score=None, error=NO_PREDICTION))


@shared_task(
    name="infer:job_score",
    bind=True,
    default_retry_delay=30,
    max_retries=3,
    soft_time_limit=10000,
)
def job_ct_score(self, job_id):
    """Score the articles that have text (text_status == ok). Every other
    article gets score null + error = its text_status; an article the model
    cannot score gets score null + error no_prediction (shared triage guard, §2).
    Nothing on this path writes 0 for "no text"."""
    try:
        db = SessionLocal()
        job = crud.job.get(db=db, id=job_id)

        scorable = []
        for article in job.article_set:
            text_status = legacy_text_status(article.text_status, article.text)
            if text_status == TEXT_OK:
                scorable.append(article)
            else:
                crud.article.update(
                    db=db, db_obj=article,
                    obj_in=ArticleUpdate(score=None, text_status=text_status, error=text_status),
                )
        if not scorable:
            log.warning("job %s: no article with text, nothing to score", job_id)
            return job_id

        article_df = pd.DataFrame({
            "pmid": [article.pmid for article in scorable],
            TEXT_COL: [str(article.text) for article in scorable],
        })
        try:
            predictions, _ = get_tagger().predict(article_df)
            scores = predictions.numpy().flatten().tolist()
        except Exception as e:
            log.exception("job %s: inference failed for %d articles: %s", job_id, len(scorable), e)
            for article in scorable:
                _mark_no_prediction(db, article)
            return job_id
        if len(scores) != len(scorable):
            log.error("job %s: model returned %d predictions for %d texts", job_id, len(scores), len(scorable))
            for article in scorable:
                _mark_no_prediction(db, article)
            return job_id

        for article, pred_score in zip(scorable, scores):
            if not _usable_score(pred_score):
                log.error("job %s pmid %s: no usable score from the model: %r", job_id, article.pmid, pred_score)
                _mark_no_prediction(db, article)
                continue
            article_update = ArticleUpdate(score=float(pred_score), error=None)
            crud.article.update(db=db, db_obj=article, obj_in=article_update)

    finally:
        db.close()
    return job_id
