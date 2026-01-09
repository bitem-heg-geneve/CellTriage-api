import asyncio
import aiohttp
from app.db.session import SessionLocal
from app import crud
import logging
from celery import shared_task
from app.schemas.article import ArticleUpdate
from app.schemas.job import JobUpdate
from app.tasks import sibils

# Shared Sibils handler to enforce max concurency for api requests.
sibils = sibils.sibils_handler


def get_text(pmids, use_fulltext):
    medl_res = sibils.fetch(ids=pmids, col="medline")
    res = {}
    pmcids = []

    for k, v in medl_res.items():
        pmcid = v["document"]["pmcid"]
        if pmcid != "":
            pmcids.append(pmcid)
        art = {
            "pmcid": pmcid,
            "entrez_date": v["document"]["entrez_date"],
            "text_source": "abstract",
            "text": v["document"]["title"] + " " + v["document"]["abstract"],
        }
        res[int(k)] = art

    if use_fulltext:
        # overwrite text with fulltext is availabe
        pmc_res = sibils.fetch(ids=pmcids, col="pmc")
        for r in res.values():
            pmcid = r["pmcid"]
            if pmcid in pmc_res.keys():
                art_pmc = pmc_res[pmcid]
                text = " ".join([s["sentence"] for s in art_pmc["sentences"]])
                if text:
                    r["text"] = text
                    r["text_source"] = "fulltext"
    return res


@shared_task(
    name="ingress:job_text",
    bind=True,
    autoretry_for=(asyncio.TimeoutError, aiohttp.ClientError, OSError, ConnectionError),
    retry_backoff=30,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=5,
    soft_time_limit=10000,
)
def job_text(self, job_id):
    db = SessionLocal()
    job = crud.job.get(db=db, id=job_id)
    try:
        article_set = [article for article in job.article_set]
        pmids = [article.pmid for article in article_set]
        res_dict = get_text(pmids, use_fulltext=job.use_fulltext)
        for article in article_set:
            # TODO
            if article.pmid in res_dict.keys():
                res = res_dict[article.pmid]
                if "text" not in res:
                    res["text"] = ""
                article_update = ArticleUpdate(**res)
                crud.article.update(db=db, db_obj=article, obj_in=article_update)

    except Exception as e:
        logging.info(f"job {job.id} status = failed")
        job_update = JobUpdate(status="failed")
        crud.job.update(db=db, db_obj=job, obj_in=job_update)

    db.close()
    return job_id
