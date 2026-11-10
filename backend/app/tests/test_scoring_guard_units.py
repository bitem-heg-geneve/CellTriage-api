"""Pure-function tests of the shared triage guard: no network, no database, no torch.

    cd backend/app && python -m pytest tests/test_scoring_guard_units.py
"""
import pytest

from app.services.job_summary import (
    NO_PREDICTION,
    TEXT_EMPTY,
    TEXT_FETCH_ERROR,
    TEXT_MISSING,
    TEXT_OK,
    legacy_text_status,
    summarize,
)
from app.services.sibils_api import (
    FetchResult,
    SibilsFetchError,
    parse_fetch_response,
    parse_missing_ids,
    sibils_base_url,
)


# --- §5 configuration -------------------------------------------------------

def test_sibils_base_url_strips_trailing_slash():
    assert sibils_base_url("https://biodiversitypmc.sibils.org/api") == "https://biodiversitypmc.sibils.org/api"
    assert sibils_base_url("https://biodiversitypmc.sibils.org/api/") == "https://biodiversitypmc.sibils.org/api"
    assert sibils_base_url(" http://sibils:8080/api// ") == "http://sibils:8080/api"


def test_sibils_base_url_refuses_missing_or_non_http():
    for bad in ("", None, "   ", "sibils.text-analytics.ch/api", "ftp://x"):
        with pytest.raises(RuntimeError):
            sibils_base_url(bad)


# --- §1 response parsing ----------------------------------------------------

def test_parse_missing_ids():
    assert parse_missing_ids("missing ids:999999999") == ["999999999"]
    assert parse_missing_ids("missing ids: 1, 2,3") == ["1", "2", "3"]
    assert parse_missing_ids("") == []
    assert parse_missing_ids(None) == []
    assert parse_missing_ids("something else") == []


def record(pmid, title="A title", abstract="An abstract."):
    return {"_id": str(pmid), "document": {"title": title, "abstract": abstract, "pmcid": ""}}


def test_parse_fetch_response_found_missing_and_versions():
    body = {
        "success": True,
        "warning": "missing ids:999999999",
        "sibils_version": "v5.1.5.10",
        "collection_version": "med_r1_v5.10",
        "sibils_article_set": [record(42967740), record(42989505)],
    }
    res = parse_fetch_response(body, [42967740, 42989505, 999999999])
    assert set(res.found) == {"42967740", "42989505"}
    assert res.missing == ["999999999"]
    assert (res.sibils_version, res.collection_version) == ("v5.1.5.10", "med_r1_v5.10")
    assert res.warning == "missing ids:999999999"


def test_parse_fetch_response_unknown_pmid_only():
    body = {"success": True, "warning": "missing ids:999999999", "sibils_article_set": []}
    res = parse_fetch_response(body, ["999999999"])
    assert res.found == {} and res.missing == ["999999999"]
    assert res.sibils_version is None


def test_parse_fetch_response_set_difference_is_authoritative():
    # No warning at all, record absent: still missing.
    res = parse_fetch_response({"success": True, "sibils_article_set": [record(1)]}, [1, 2])
    assert res.missing == ["2"] and res.warning is None


def test_parse_fetch_response_rejects_non_answers():
    with pytest.raises(SibilsFetchError):
        parse_fetch_response([], [1])
    with pytest.raises(SibilsFetchError):
        parse_fetch_response("<html>502</html>", [1])
    with pytest.raises(SibilsFetchError):
        parse_fetch_response({"success": False, "error": "boom"}, [1])


def test_fetch_result_merge_keeps_first_versions_and_dedups_warnings():
    a = FetchResult(found={"1": {}}, missing=["2"], warnings=["w1"], sibils_version="v1", collection_version="c1")
    b = FetchResult(found={"3": {}}, missing=["2", "4"], warnings=["w1", "w2"], sibils_version="v2")
    a.merge(b)
    assert set(a.found) == {"1", "3"} and a.missing == ["2", "4"]
    assert a.warnings == ["w1", "w2"] and a.warning == "w1; w2"
    assert (a.sibils_version, a.collection_version) == ("v1", "c1")


# --- §3 job summary ---------------------------------------------------------

def rows(*specs):
    """specs: 'scored', 'missing', 'empty', 'fetch_error', 'no_prediction', 'pending',
    'legacy_zero' (pre-guard row: no text_status, score 0.0), 'legacy_pending'."""
    out = []
    for spec in specs:
        if spec == "scored":
            out.append((TEXT_OK, 0.87, None))
        elif spec in (TEXT_MISSING, TEXT_EMPTY, TEXT_FETCH_ERROR):
            out.append((spec, None, spec))
        elif spec == "no_prediction":
            out.append((TEXT_OK, None, NO_PREDICTION))
        elif spec == "pending":
            out.append((TEXT_OK, None, None))
        elif spec == "legacy_zero":
            out.append((None, 0.0, None))
        elif spec == "legacy_pending":
            out.append((None, None, None))
        else:
            raise ValueError(spec)
    return out


def test_done_when_everything_scored():
    s = summarize(rows("scored", "scored"))
    assert (s.status, s.n_submitted, s.n_scored, s.n_no_text, s.error) == ("done", 2, 2, 0, None)


def test_partial_with_one_missing():
    s = summarize(rows("scored", "scored", "missing"))
    assert s.status == "partial"
    assert (s.n_submitted, s.n_scored, s.n_no_text, s.n_fetch_error, s.n_no_prediction) == (3, 2, 1, 0, 0)
    assert s.error is None


def test_counts_by_kind():
    s = summarize(rows("scored", "missing", "empty", "fetch_error", "no_prediction"))
    assert (s.n_scored, s.n_no_text, s.n_fetch_error, s.n_no_prediction) == (1, 2, 1, 1)
    assert s.status == "partial"


def test_failed_when_nothing_scored():
    s = summarize(rows("missing", "empty"))
    assert (s.status, s.error, s.n_no_text) == ("failed", "sibils_fetch", 2)
    s = summarize(rows("fetch_error", "fetch_error"))
    assert (s.status, s.error, s.n_fetch_error) == ("failed", "sibils_fetch", 2)
    s = summarize(rows("no_prediction", "no_prediction"))
    assert (s.status, s.error, s.n_no_prediction) == ("failed", "no_prediction", 2)


def test_failed_when_more_than_half_without_text_in_a_big_job():
    s = summarize(rows(*(["scored"] * 9 + ["missing"] * 11)))
    assert (s.status, s.error) == ("failed", "sibils_fetch")
    # Same share in a small job is partial: a few odd PMIDs, not an endpoint problem.
    s = summarize(rows(*(["scored"] * 4 + ["missing"] * 6)))
    assert s.status == "partial"
    # Exactly half of a big job is still partial.
    s = summarize(rows(*(["scored"] * 10 + ["fetch_error"] * 10)))
    assert s.status == "partial"


def test_pending_and_in_progress():
    assert summarize(rows("pending", "pending")).status == "pending"
    assert summarize(rows("legacy_pending")).status == "pending"
    assert summarize(rows("scored", "pending")).status == "in progress"
    assert summarize(rows("missing", "pending")).status == "in progress"
    assert summarize([]).status == "pending"


def test_final_turns_leftover_pending_into_no_prediction():
    s = summarize(rows("scored", "pending"), final=True)
    assert (s.status, s.n_scored, s.n_no_prediction, s.n_pending) == ("partial", 1, 1, 0)
    assert summarize([], final=True).status == "done"


def test_legacy_rows_without_text_status():
    # A pre-guard 0.0 is still a stored score; the guard cannot rewrite history.
    s = summarize(rows("legacy_zero", "scored"))
    assert (s.status, s.n_scored) == ("done", 2)
    assert legacy_text_status(None, "some text") == TEXT_OK
    assert legacy_text_status(None, "") == TEXT_MISSING
    assert legacy_text_status(None, None) == TEXT_MISSING
    assert legacy_text_status(TEXT_EMPTY, "ignored") == TEXT_EMPTY


def test_degenerate_scores_fail_a_finished_job():
    from app.services.job_summary import (DEGENERATE_MIN_SCORED, DEGENERATE_SCORES, JobSummary,
                                          degenerate_scores, with_degenerate_check)
    n = DEGENERATE_MIN_SCORED
    done = JobSummary(n, n, 0, 0, 0, 0, "done", None)
    assert degenerate_scores([0.0] * n) and degenerate_scores([0.5] * n)
    assert not degenerate_scores([0.0] * (n - 1))            # too few to judge
    assert not degenerate_scores([0.0] * (n - 1) + [0.01])   # one different value is enough
    out = with_degenerate_check(done, [0.0] * n)
    assert out.status == "failed" and out.error == DEGENERATE_SCORES and out.n_scored == n
    assert with_degenerate_check(done, [i / n for i in range(n)]) == done
    running = JobSummary(n, 0, 0, 0, 0, n, "in progress", None)
    assert with_degenerate_check(running, [0.0] * n) == running
