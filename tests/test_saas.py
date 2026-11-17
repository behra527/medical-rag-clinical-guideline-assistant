"""Product-facing contract tests; all tests are offline and use fake embeddings."""
from pathlib import Path
from fastapi.testclient import TestClient
import app.api as routes
from .conftest import sample_pdf


def test_home_and_assets_and_health(rag, monkeypatch):
    monkeypatch.setattr(routes, 'get_rag', lambda: rag)
    api = TestClient(routes.app)
    assert api.get('/').status_code == 200
    assert 'Clinical Evidence Intelligence' in api.get('/').text
    assert api.get('/assets/styles.css').status_code == 200
    assert api.get('/assets/app.js').status_code == 200
    status = api.get('/health').json()
    assert status['ready'] and status['documents'] == 1
    assert status['chunks'] > 0
    settings = api.get('/api/system').json()
    assert settings['embedding_model'] == rag.settings.embed_model
    assert not ('api_key' in settings)


def test_source_pdf_and_safe_cleanup(rag, monkeypatch):
    monkeypatch.setattr(routes, 'get_rag', lambda: rag)
    api = TestClient(routes.app)
    doc_id = rag.store.list_documents()[0]['id']
    result = api.get(f'/documents/{doc_id}/pdf')
    assert result.status_code == 200
    assert result.content == sample_pdf()
    assert api.get('/documents/../../app/config.py/pdf').status_code in (404, 405)
    assert api.delete(f'/documents/{doc_id}').status_code == 200
    assert not (Path(rag.settings.pdf_dir) / f'{doc_id}.pdf').exists()
    assert api.get(f'/documents/{doc_id}/pdf').status_code == 404


def test_official_catalog_import_and_rights_check(rag, monkeypatch):
    monkeypatch.setattr(routes, 'get_rag', lambda: rag)
    monkeypatch.setattr(routes, 'download_pdf', lambda url, cap: sample_pdf())
    api = TestClient(routes.app)
    entries = api.get('/api/catalog').json()
    assert len(entries) >= 2
    assert any(e['publisher'] == 'World Health Organization' for e in entries)
    no_rights = api.post('/api/catalog/import', json={'guideline_id': 'who-hypertension-2021', 'rights_confirmed': False})
    assert no_rights.status_code == 400
    bad_id = api.post('/api/catalog/import', json={'guideline_id': 'madeup', 'rights_confirmed': True})
    assert bad_id.status_code == 404
    duplicate = api.post('/api/catalog/import', json={'guideline_id': 'who-hypertension-2021', 'rights_confirmed': True})
    assert duplicate.status_code == 200
    assert duplicate.json()['status'] == 'already_indexed'


def test_feedback_tracks_votes_without_storing_queries(rag, monkeypatch):
    monkeypatch.setattr(routes, 'get_rag', lambda: rag)
    api = TestClient(routes.app)
    assert api.post('/api/feedback', json={'vote':'helpful'}).status_code == 200
    assert api.post('/api/feedback', json={'vote':'not_helpful'}).status_code == 200
    assert api.post('/api/feedback', json={'vote':'maybe'}).status_code == 422
    assert api.get('/api/feedback/stats').json() == {'helpful':1,'not_helpful':1}


def test_chat_rate_limit(rag, monkeypatch):
    monkeypatch.setattr(routes, 'get_rag', lambda: rag)
    monkeypatch.setattr(routes, 'CHAT_LIMIT', 2)
    api = TestClient(routes.app)
    body = {'question': 'Diagnosing blood pressure in adults?'}
    assert api.post('/chat', json=body).status_code == 200
    assert api.post('/chat', json=body).status_code == 200
    assert api.post('/chat', json=body).status_code == 429


def test_admin_session_cookie_replaces_header(rag, monkeypatch):
    from dataclasses import replace
    rag.settings = replace(rag.settings, admin_api_key='supersecret')
    monkeypatch.setattr(routes, 'get_rag', lambda: rag)
    api = TestClient(routes.app)
    assert api.post('/documents/upload', files={'file': ('x.pdf', sample_pdf(), 'application/pdf')}).status_code == 403
    assert api.post('/api/admin/session', json={'key': 'nope'}).status_code == 403
    assert api.post('/api/admin/session', json={'key': 'supersecret'}).status_code == 200
    assert api.get('/api/admin/session').json()['active'] is True
    assert 'vh_admin' not in api.post('/api/admin/session', json={'key': 'supersecret'}).json()
    uploaded = api.post('/documents/upload', files={'file': ('demo.pdf', sample_pdf(), 'application/pdf')})
    assert uploaded.status_code == 200
    assert api.delete('/api/admin/session').status_code == 200
    assert api.post('/documents/upload', files={'file': ('demo.pdf', sample_pdf(), 'application/pdf')}).status_code == 403


def test_background_catalog_import(rag, monkeypatch):
    import time
    monkeypatch.setattr(routes, 'get_rag', lambda: rag)
    monkeypatch.setattr(routes, 'download_pdf', lambda url, cap: sample_pdf())
    api = TestClient(routes.app)
    started = api.post('/api/catalog/import', json={'guideline_id': 'who-hypertension-2021', 'rights_confirmed': True, 'background': True})
    assert started.status_code == 200 and started.json()['status'] == 'running'
    job = {'status': 'running'}
    for _ in range(40):
        job = api.get(f"/api/jobs/{started.json()['job_id']}").json()
        if job['status'] != 'running':
            break
        time.sleep(0.05)
    assert job['status'] == 'done'
    assert job['result']['status'] in ('indexed', 'already_indexed')


def test_admin_key_and_production_enforced(rag, monkeypatch):
    from dataclasses import replace
    rag.settings = replace(rag.settings, admin_api_key='supersecret')
    monkeypatch.setattr(routes, 'get_rag', lambda: rag)
    api = TestClient(routes.app)
    assert api.post('/api/catalog/import', json={'guideline_id':'who-hypertension-2021','rights_confirmed':True}).status_code == 403
    assert api.post('/documents/upload', files={'file':('x.pdf', sample_pdf(), 'application/pdf')}).status_code == 403
    assert api.get('/api/feedback/stats').status_code == 403
    rag.settings = replace(rag.settings, admin_api_key='')
    monkeypatch.setenv('APP_ENV', 'production')
    assert api.post('/documents/upload', files={'file':('x.pdf', sample_pdf(), 'application/pdf')}).status_code == 503
