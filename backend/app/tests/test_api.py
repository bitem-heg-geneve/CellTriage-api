def test_docs_redirect(client):
    r = client.get('/', follow_redirects=True)
    assert r.status_code == 200


def test_create_job_returns_id(client):
    r = client.post(
        '/api/v1/jobs',
        json={'article_set': [{'pmid': 36754106}], 'use_fulltext': False},
    )
    assert r.status_code == 200
    data = r.json()
    assert 'id' in data
    assert data['status'] == 'pending'


def test_get_job_not_found(client):
    r = client.get('/api/v1/jobs/999999')
    assert r.status_code == 404


def test_get_job_status(client):
    r = client.post(
        '/api/v1/jobs',
        json={'article_set': [{'pmid': 36754106}], 'use_fulltext': False},
    )
    job_id = r.json()['id']
    s = client.get(f'/api/v1/jobs/{job_id}/status')
    assert s.status_code == 200
    assert 'status' in s.json()
