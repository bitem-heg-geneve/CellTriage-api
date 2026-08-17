import httpx

BASE = "http://localhost:8001"


def test_docs():
    r = httpx.get(f"{BASE}/", follow_redirects=True)
    assert r.status_code == 200


def test_create_job_returns_id():
    r = httpx.post(
        f"{BASE}/api/v1/jobs",
        json={"article_set": [{"pmid": 36754106}], "use_fulltext": False},
    )
    assert r.status_code == 200
    data = r.json()
    assert "id" in data
    assert data["status"] == "pending"


def test_create_job_empty_set():
    r = httpx.post(f"{BASE}/api/v1/jobs", json={"article_set": [], "use_fulltext": False})
    assert r.status_code in (200, 400, 422)


def test_get_job_not_found():
    r = httpx.get(f"{BASE}/api/v1/jobs/999999")
    assert r.status_code == 404
