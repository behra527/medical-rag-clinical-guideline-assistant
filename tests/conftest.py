"""Small synthetic PDFs; not offered as actual medical clinical guidelines."""
from functools import lru_cache
import numpy as np
import fitz
import pytest
from app.config import Settings
from app.rag import MedicalRAG
from app.store import KnowledgeStore


@lru_cache(maxsize=1)
def sample_pdf() -> bytes:
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((55, 65), '1.2 Diagnosing hypertension', fontsize=14)
    p1.insert_textbox(fitz.Rect(55, 100, 540, 250),
        'For confirmation of high blood pressure in adults, use ambulatory blood pressure monitoring (ABPM). '
        'If ambulatory monitoring is not suitable, consider home blood pressure monitoring (HBPM). '
        'Check that the blood pressure device is correctly validated.', fontsize=12)
    p2 = doc.new_page()
    p2.insert_text((55, 65), '1.4 Treating and monitoring hypertension', fontsize=14)
    p2.insert_textbox(fitz.Rect(55, 100, 540, 250),
        'For adults aged under 80 the home blood pressure treatment target is below 135/85 mmHg. '
        'In older adults aged 80 or above, guidelines may use a different treatment target. '
        'Review treatment based on individual needs and frailty.', fontsize=12)
    result = doc.tobytes()
    doc.close()
    return result


class FakeOpenRouter:
    """Deliberately simplified semantic vectors to test orchestration, not model quality."""
    def __init__(self, answer='Based on the guideline, ABPM is recommended [S1].'):
        self.answer = answer
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        vectors = []
        for t in texts:
            t = t.lower()
            if 'asthma' in t or 'inhaler' in t:
                vectors.append([0., 1., 0.])
            elif 'car' in t or 'engine' in t:
                vectors.append([0., 0., 1.])
            else:
                vectors.append([1., 0., 0.])
        return np.array(vectors, dtype=np.float32)

    def complete(self, messages, **kwargs):
        self.calls += 1
        return self.answer

    def complete_stream(self, messages, **kwargs):
        text = self.complete(messages, **kwargs)
        step = 12
        for start in range(0, len(text), step):
            yield text[start:start + step]


@pytest.fixture(autouse=True)
def _reset_request_limits():
    from app import api as routes
    routes.reset_rate_limit()
    yield
    routes.reset_rate_limit()


@pytest.fixture
def rag(tmp_path):
    settings = Settings(api_key='fake-key', admin_api_key='', db_path=str(tmp_path / 'index.db'),
                        pdf_dir=str(tmp_path / 'pdfs'),
                        min_similarity=.5, top_k=4, chunk_words=55, chunk_overlap=10)
    client = FakeOpenRouter()
    system = MedicalRAG(settings, client=client, store=KnowledgeStore(settings.db_path))
    system.ingest(sample_pdf(), 'demo.pdf', title='NICE NG136 — Test Fixture', source_url='https://example.org/guideline')
    return system
