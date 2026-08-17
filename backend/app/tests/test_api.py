import httpx

BASE = "http://localhost:8001"
TIMEOUT = 30.0


def test_healthz():
    r = httpx.get(f"{BASE}/api/v1/jobs/0/status", timeout=TIMEOUT)
    # 404 means the server is up and routing correctly
    assert r.status_code in (200, 404)


def test_create_job_returns_id():
    r = httpx.post(
        f"{BASE}/api/v1/jobs",
        json={"article_set": [{"pmid": 36754106}], "use_fulltext": False},
        timeout=TIMEOUT,
    )
    assert r.status_code == 200
    data = r.json()
    assert "id" in data
    assert data["status"] == "pending"


def test_create_job_empty_set():
    r = httpx.post(
        f"{BASE}/api/v1/jobs",
        json={"article_set": [], "use_fulltext": False},
        timeout=TIMEOUT,
    )
    assert r.status_code in (200, 400, 422)


def test_get_job_not_found():
    r = httpx.get(f"{BASE}/api/v1/jobs/999999", timeout=TIMEOUT)
    assert r.status_code == 404
