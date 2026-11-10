"""Job-level outcome of the shared triage guard (§3).

Pure functions over the per-article (text_status, score, error) triples, so the
API, the worker and the tests agree on one definition of n_scored / n_no_text /
n_fetch_error / n_no_prediction and of the job status:

- done        every article is scored;
- partial     some are not, the scored ones are valid;
- failed      none is scored, or no-text + fetch-error exceed FAILED_NO_TEXT_SHARE
              of a job of at least FAILED_MIN_ARTICLES (an endpoint or collection
              problem, not a few odd PMIDs); `error` says which;
- pending / in progress   not every article has an outcome yet.

Ported from DisTriage's services/job_summary.py (be2f802); CellTriage has no
per-stage status columns, so an article's outcome is read from text_status,
score and error alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

TEXT_OK = "ok"
TEXT_MISSING = "missing"
TEXT_EMPTY = "empty"
TEXT_FETCH_ERROR = "fetch_error"
NO_PREDICTION = "no_prediction"

STATUS_PENDING = "pending"
STATUS_RUNNING = "in progress"
STATUS_DONE = "done"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"

FAILED_NO_TEXT_SHARE = 0.5
FAILED_MIN_ARTICLES = 20

Row = Tuple[Optional[str], Optional[float], Optional[str]]


@dataclass(frozen=True)
class JobSummary:
    n_submitted: int
    n_scored: int
    n_no_text: int
    n_fetch_error: int
    n_no_prediction: int
    n_pending: int
    status: str
    error: Optional[str]


def legacy_text_status(text_status: Optional[str], text: Optional[str]) -> str:
    """Articles written before the guard have no text_status; derive one from the text."""
    if text_status:
        return text_status
    return TEXT_OK if (text or "").strip() else TEXT_MISSING


def article_row(article) -> Row:
    """The triple `summarize` needs, from an ORM Article (or anything with the attributes)."""
    return (article.text_status, article.score, article.error)


def summarize(rows: Iterable[Row], final: bool = False) -> JobSummary:
    """rows: (text_status, score, error) per article.

    `final=True` is for the end of the pipeline: an article that still has
    neither a score nor a reason counts as no_prediction instead of pending."""
    n = scored = no_text = fetch_error = no_prediction = pending = 0
    for text_status, score, error in rows:
        n += 1
        if text_status in (TEXT_MISSING, TEXT_EMPTY):
            no_text += 1
        elif text_status == TEXT_FETCH_ERROR:
            fetch_error += 1
        elif error == NO_PREDICTION:
            no_prediction += 1
        elif score is not None:
            scored += 1
        elif final:
            no_prediction += 1
        else:
            pending += 1

    error = None
    if n == 0:
        status = STATUS_DONE if final else STATUS_PENDING
    elif pending == n:
        status = STATUS_PENDING
    elif pending > 0:
        status = STATUS_RUNNING
    elif scored == n:
        status = STATUS_DONE
    elif scored == 0 or (
        n >= FAILED_MIN_ARTICLES and (no_text + fetch_error) > FAILED_NO_TEXT_SHARE * n
    ):
        status = STATUS_FAILED
        error = "sibils_fetch" if (no_text + fetch_error) >= no_prediction else NO_PREDICTION
    else:
        status = STATUS_PARTIAL
    return JobSummary(n, scored, no_text, fetch_error, no_prediction, pending, status, error)
