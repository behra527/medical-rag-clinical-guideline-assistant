from fastapi.testclient import TestClient
import app.api as routes
from .conftest import sample_pdf


def test_http_endpoints_with_fake_provider(rag, monkeypatch):
    monkeypatch.setattr(routes, 'get_rag', lambda: rag)
    client = TestClient(routes.app)
    health = client.get('/health')
    assert health.status_code == 200
    assert health.json()['documents'] == 1
    assert client.get('/documents').json()[0]['title'].startswith('NICE NG136')
    assert client.post('/retrieve', json={'question': 'Diagnosing blood pressure in adults?'}).json()['results']
    ans = client.post('/chat', json={'question': 'Diagnosing blood pressure in adults?'})
    assert ans.status_code == 200 and '[S1 p.' in ans.json()['answer']
    follow = client.post('/chat', json={'question': 'What about adults aged 80?', 'history': [{'question': 'How is blood pressure treated?'}]})
    assert follow.status_code == 200 and follow.json()['follow_up'] is True
    streamed = client.post('/chat', json={'question': 'Diagnosing blood pressure in adults?', 'stream': True})
    assert streamed.status_code == 200
    assert 'text/event-stream' in streamed.headers['content-type']
    assert 'event: done' in streamed.text and '[S1 p.' in streamed.text
    invalid = client.post('/chat', json={'question':'  '})
    assert invalid.status_code == 400
    assert client.post('/chat', json={'question':'How do I fix my car engine?'}).json()['abstained']
    upload = client.post('/documents/upload', files={'file':('demo.pdf',sample_pdf(),'application/pdf')})
    assert upload.status_code == 200 and upload.json()['status'] == 'already_indexed'
    no_url = client.post('/documents/upload', data={'source_url':'file:///etc/passwd'}, files={'file':('x.pdf',sample_pdf(),'application/pdf')})
    assert no_url.status_code == 400
    ev = client.post('/evaluation/run', json={'cases':[{'question': 'Diagnosing blood pressure?', 'gold_section':'1.2'}], 'include_generation':False})
    assert ev.status_code == 200
    doc_id = client.get('/documents').json()[0]['id']
    assert client.delete(f'/documents/{doc_id}').json()['deleted']
    assert client.get('/health').json()['documents'] == 0
