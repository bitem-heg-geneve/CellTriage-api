import asyncio
import aiohttp
from app.db.session import SessionLocal
from app import crud
import logging
from celery import shared_task
from celery.utils.log import get_task_logger
from app.schemas.article import ArticleUpdate
from app.schemas.job import JobUpdate
from app.tasks import sibils
from app.core.config import settings
from app.services.job_summary import TEXT_EMPTY, TEXT_FETCH_ERROR, TEXT_MISSING, TEXT_OK
from app.services.sibils_api import FetchResult, SibilsFetchError

# Shared Sibils handler to enforce max concurency for api requests.
sibils = sibils.sibils_handler

log = get_task_logger(__name__)


def get_text(pmids, use_fulltext):
    """Title + abstract (or PMC full text) per PMID SIBiLS returned, and the
    FetchResult saying which PMIDs it did not return and which instance answered.
    Raises SibilsFetchError when the MEDLINE fetch fails after retries."""
    medl_res = sibils.fetch(ids=pmids, col="medline")
    res = {}
    pmcids = []

    for k, v in medl_res.found.items():
        doc = v.get("document") or {}
        pmcid = doc.get("pmcid") or ""
        if pmcid != "":
            pmcids.append(pmcid)
        art = {
            "pmcid": pmcid,
            "entrez_date": doc.get("entrez_date"),
            "text_source": "abstract",
            "text": ((doc.get("title") or "") + " " + (doc.get("abstract") or "")).strip(),
        }
        res[int(k)] = art

    if use_fulltext and pmcids:
        # overwrite text with fulltext if available; full text is a bonus, the
        # abstract is enough to score, so a PMC failure is not a fetch_error.
        try:
            pmc_res = sibils.fetch(ids=pmcids, col="pmc")
        except SibilsFetchError as exc:
            log.warning("PMC fetch failed, scoring on abstracts: %s", exc)
            pmc_res = FetchResult()
        if pmc_res.warning:
            log.info("PMC fetch: %s", pmc_res.warning)
        for r in res.values():
            pmcid = r["pmcid"]
            if pmcid in pmc_res.found:
                art_pmc = pmc_res.found[pmcid]
                text = " ".join([s["sentence"] for s in art_pmc.get("sentences") or []])
                if text:
                    r["text"] = text
                    r["text_source"] = "fulltext"
    return res, medl_res


def _mark_no_text(db, article, text_status):
    """§1/§2: the article has no text; text_status and error say why, never ""."""
    crud.article.update(
        db=db, db_obj=article,
        obj_in=ArticleUpdate(text=None, score=None, text_status=text_status, error=text_status),
    )


def _record_fetch_meta(db, job, fetch, n_requested):
    """Log the SIBiLS instance once per job (INFO) and keep its warnings on the job."""
    log.info(
        "job %s: SIBiLS %s, collection %s, url %s: %d of %d PMIDs returned",
        job.id, fetch.sibils_version, fetch.collection_version, settings.SIBILS_URL,
        len(fetch.found), n_requested,
    )
    if fetch.warning:
        log.warning("job %s: SIBiLS warning: %s", job.id, fetch.warning)
    job_update = JobUpdate(
        sibils_version=fetch.sibils_version,
        collection_version=fetch.collection_version,
        sibils_warning=(fetch.warning or "")[:2000] or None,
    )
    crud.job.update(db=db, db_obj=job, obj_in=job_update)


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
    """Fetch text for every article of the job and record, per article, what
    came back: text_status ok | missing | empty | fetch_error. Nothing on this
    path stores text "" as if it had been fetched (shared triage guard, §1)."""
    db = SessionLocal()
    job = crud.job.get(db=db, id=job_id)
    article_set = [article for article in job.article_set]
    pmids = [article.pmid for article in article_set]
    try:
        try:
            res_dict, fetch = get_text(pmids, use_fulltext=job.use_fulltext)
        except SibilsFetchError as e:
            log.error("job %s: SIBiLS fetch failed for %d PMIDs: %s", job.id, len(pmids), e)
            for article in article_set:
                _mark_no_text(db, article, TEXT_FETCH_ERROR)
            crud.job.update(db=db, db_obj=job, obj_in=JobUpdate(sibils_warning=str(e)[:2000]))
            return job_id

        _record_fetch_meta(db, job, fetch, len(pmids))
        for article in article_set:
            res = res_dict.get(article.pmid)
            if res is None:
                _mark_no_text(db, article, TEXT_MISSING)
            elif not res["text"]:
                crud.article.update(
                    db=db, db_obj=article,
                    obj_in=ArticleUpdate(**{**res, "text": None}, score=None,
                                         text_status=TEXT_EMPTY, error=TEXT_EMPTY),
                )
            else:
                article_update = ArticleUpdate(**res, score=None, text_status=TEXT_OK, error=None)
                crud.article.update(db=db, db_obj=article, obj_in=article_update)

    except Exception as e:
        # Anything else (a malformed record, the database): the articles not
        # yet recorded get fetch_error so the job ends failed/partial, not
        # "done" with empty text.
        log.exception("job %s: text stage failed: %s", job.id, e)
        db.rollback()
        for article in article_set:
            if not article.text_status:
                _mark_no_text(db, article, TEXT_FETCH_ERROR)
        crud.job.update(db=db, db_obj=job, obj_in=JobUpdate(sibils_warning=f"text stage error: {e}"[:2000]))

    finally:
        db.close()
    return job_id
