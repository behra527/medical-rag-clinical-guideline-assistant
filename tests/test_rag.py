import pytest
from app.pdf_ingest import extract_chunks
from app.rag import ABSTENTION, MedicalRAG
from app.store import KnowledgeStore
from app.config import Settings
from .conftest import FakeOpenRouter, sample_pdf


def test_pdf_parsing_preserves_pages_and_sections():
    chunks = extract_chunks(sample_pdf(), words_per_chunk=55, overlap=10)
    assert len(chunks) >= 2
    assert any(c.page == 1 and c.section.startswith('1.2') and 'blood pressure' in c.text.lower() for c in chunks)
    assert any(c.page == 2 and c.section.startswith('1.4') for c in chunks)


def test_invalid_pdfs_raise():
    with pytest.raises(ValueError, match='Invalid file'):
        extract_chunks(b'not-pdf')
    with pytest.raises(ValueError, match='extractable text'):
        blank = __import__('fitz').open()
        blank.new_page()
        extract_chunks(blank.tobytes())


def test_doc_deduplication(rag):
    before = len(rag.store.list_documents())
    result = rag.ingest(sample_pdf(), 'demo.pdf')
    assert result['status'] == 'already_indexed'
    assert len(rag.store.list_documents()) == before


def test_empty_invalid_questions(rag):
    for q in ('', '  ', '??', 'x' * 1001, '123'):
        with pytest.raises(ValueError, match='meaningful medical question'):
            rag.chat(q)


def test_retrieve_expected_section_and_credible_metadata(rag):
    chunks = rag.retrieve('How to diagnose blood pressure in adults?')
    assert chunks
    assert chunks[0]['section'].startswith('1.2')
    assert chunks[0]['page'] == 1
    assert chunks[0]['source_url'] == 'https://example.org/guideline'
    assert 0 <= chunks[0]['similarity'] <= 1


def test_unsupported_query_abstains_without_llm_call(rag):
    calls = rag.client.calls
    response = rag.chat('How do I fix my car engine?')
    assert response['abstained'] is True
    assert response['reason'] == 'insufficient_evidence'
    assert not response['sources']
    # Only query embedding, never LLM generation.
    assert rag.client.calls == calls + 1


def test_valid_grounded_answer(rag):
    response = rag.chat('How do I diagnose blood pressure?')
    assert not response['abstained']
    assert response['sources'][0]['id'] == 'S1'
    assert response['sources'][0]['page'] in [1, 2]
    assert '[S1 p.' in response['answer'] and '§' in response['answer']


def test_hallucinated_citation_is_rejected(rag):
    rag.client.answer = 'No problem, take this [S999].'
    response = rag.chat('How do I diagnose blood pressure?')
    assert response['abstained']
    assert response['reason'] == 'citation_validation_failed'
    rag.client.answer = 'No citation here.'
    assert rag.chat('How do I diagnose blood pressure?')['abstained']


def test_rerank_prefers_passage_with_query_number(rag):
    rows = rag.retrieve('treatment target for adults aged 80')
    assert rows
    assert rows[0]['section'].startswith('1.4')
    assert 'rerank_score' in rows[0]


def test_unsupported_claim_sentence_is_removed(rag):
    rag.client.answer = 'ABPM is recommended [S1]. The target is 500/500 mmHg [S1].'
    response = rag.chat('How do I diagnose blood pressure?')
    assert response['abstained'] is False
    assert '500' not in response['answer']
    assert '[S1 p.' in response['answer']
    assert response['unsupported_claims'][0]['reason'] == 'number_not_in_source'


def test_claim_with_unknown_term_abstains(rag):
    rag.client.answer = 'Use XYZQ daily [S1].'
    response = rag.chat('How do I diagnose blood pressure?')
    assert response['abstained'] is True
    assert response['reason'] == 'claim_support_failed'


def test_follow_up_merges_previous_question(rag):
    rag.client.answer = 'In older adults aged 80 the treatment target may differ [S1].'
    response = rag.chat('What about adults aged 80?', history=[{'question': 'How is blood pressure treated?'}])
    assert response['follow_up'] is True
    assert '80' in response['standalone_question']
    assert 'blood pressure' in response['standalone_question'].lower()
    assert any(source['section'].startswith('1.4') for source in response['sources'])


def test_invented_page_number_is_removed(rag):
    rag.client.answer = 'ABPM is recommended [S1]. See page 99 [S1].'
    response = rag.chat('How do I diagnose blood pressure?')
    assert 'page 99' not in response['answer']
    assert response['unsupported_claims'][0]['reason'] == 'page_not_in_source'


def test_section_metadata_filter(rag):
    document_id = rag.store.list_documents()[0]['id']
    rows = rag.retrieve('blood pressure in adults', document_id=document_id, section='1.4')
    assert rows and all(row['section'].startswith('1.4') for row in rows)
    assert rag.retrieve('blood pressure in adults', section='9.9') == []


def test_delete_document_and_clear_embedding_model(rag):
    doc = rag.store.list_documents()[0]
    assert rag.store.delete_document(doc['id'])
    assert rag.store.list_documents() == []
    assert rag.chat('How do I diagnose blood pressure?')['answer'] == ABSTENTION
    assert not rag.store.delete_document(doc['id'])


def test_inconsistent_embedding_model_blocked(rag):
    with pytest.raises(ValueError, match='mismatch'):
        rag.store.check_embedding_model('totally-different-model')
