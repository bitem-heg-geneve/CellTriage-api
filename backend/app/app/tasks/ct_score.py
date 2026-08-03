import logging
from celery import shared_task
from celery.utils.log import get_logger
from app import crud
from app.db.session import SessionLocal
from app.schemas.article import ArticleUpdate
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

def get_tagger():
    """Lazy-load CtTagger to avoid blocking worker startup."""
    global tagger
    if tagger is None and os.getenv("LOAD_CT_TAGGER", "no") == "yes":
        tagger = CtTagger(CHECKPOINT_PATH, LABELS, TEXT_COL, LM_MODEL_NAME)
    return tagger


@shared_task(
    name="infer:job_score",
    bind=True,
    default_retry_delay=30,
    max_retries=3,
    soft_time_limit=10000,
)
def job_ct_score(self, job_id):
    try:
        db = SessionLocal()
        job = crud.job.get(db=db, id=job_id)
        article_list = [article.__dict__ for article in job.article_set]
        article_df = pd.DataFrame(article_list)
        
        # Check if TEXT_COL exists in article_df
        if TEXT_COL not in article_df.columns:
            print(f"Column '{TEXT_COL}' does not exist in article_df")
            db.close()
            return job_id
    
        
        article_df[TEXT_COL] = article_df[TEXT_COL].astype(str)
        predictions, _ = get_tagger().predict(article_df)

        scores = predictions.numpy().flatten()
        for article, pred_score in zip(job.article_set, scores):
            if not article.text:
                score = 0
            else:
                score = pred_score
            article_update = ArticleUpdate(score=score)
            crud.article.update(db=db, db_obj=article, obj_in=article_update)

    finally:
        db.close()
    return job_id
