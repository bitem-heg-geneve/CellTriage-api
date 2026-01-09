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

MODEL_FP = r"/model_resources/BERT_1_1.bin"


class BERT_infer(object):
    def __init__(self, model_fp, batch_size=10):
        self.BERT_MODEL = "bert-base-uncased"
        self.label2idx = {False: 0, True: 1}
        self.model_fp = model_fp
        self.model_state_dict = torch.load(
            self.model_fp, map_location=lambda storage, loc: storage
        )
        self.model = BertForSequenceClassification.from_pretrained(
            self.BERT_MODEL, state_dict=self.model_state_dict, num_labels=2
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_seq_length = 500
        self.model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            model_max_length=self.max_seq_length,
            truncation_side="right",
        )
        # self.tokenizer.model_max_length = 500
        self.batch_size = batch_size

    def score(self, txt):
        input_ids = (
            torch.tensor(self.tokenizer.encode(txt, max_length=self.max_seq_length))
            .unsqueeze(0)
            .to(self.device)
        )  # batch size 1
        output = self.model(input_ids)
        score = round(output.logits.softmax(dim=-1).tolist()[0][1] * 100, 5)
        return score


bert_infer = BERT_infer(model_fp=MODEL_FP)


@shared_task(
    name="infer:job_score",
    bind=True,
    default_retry_delay=30,
    max_retries=3,
    soft_time_limit=10000,
)
def job_ct_score(self, job_id):
    db = SessionLocal()
    job = crud.job.get(db=db, id=job_id)

    for article in job.article_set:
        if article.text:
            score = bert_infer.score(article.text)
            article_update = ArticleUpdate(score=score)
            crud.article.update(db=db, db_obj=article, obj_in=article_update)

    db.close()
    return job_id
