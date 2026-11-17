"""FastAPI API and same-origin premium web frontend.

For public deployment: configure a strong ADMIN_API_KEY, HTTPS, request limits,
legal permission for guideline content, and independent clinical validation.
"""
from __future__ import annotations
import hmac
import json
import os
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from functools import lru_cache
from urllib.parse import urlsplit
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Header, Depends, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .catalog import CATALOG
from .config import Settings
from .openrouter import OpenRouterError
from .rag import MedicalRAG
from .evaluation import run_evaluation, sweep_retrieval, compare_prompts

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'ui' / 'web'
app = FastAPI(
    title='Clinical Guideline Intelligence', version='2.0.0',
    description='Evidence-first clinical guideline question answering; not individualized medical advice.',
)
if (WEB / 'assets').is_dir():
    app.mount('/assets', StaticFiles(directory=str(WEB / 'assets')), name='assets')

@lru_cache
def get_rag():
    return MedicalRAG(Settings())


ADMIN_COOKIE = 'vh_admin'
CHAT_LIMIT = 20
CHAT_WINDOW_SECONDS = 60
_rate_hits: dict[str, deque] = defaultdict(deque)
_sessions: dict[str, float] = {}
_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()
_state_lock = threading.Lock()


def reset_rate_limit():
    _rate_hits.clear()


def _admin_required() -> str:
    return get_rag().settings.admin_api_key or ''


def _session_active(token: str | None) -> bool:
    if not token:
        return False
    with _state_lock:
        expiry = _sessions.get(token)
        if expiry is None:
            return False
        if expiry < time.time():
            _sessions.pop(token, None)
            return False
        return True


def protect_write(request: Request, x_admin_key: str | None = Header(default=None)):
    required = _admin_required()
    if required.startswith('replace_'):
        raise HTTPException(503, 'Set a unique ADMIN_API_KEY in .env before administrative actions')
    if not required and os.getenv('APP_ENV', '').lower() == 'production':
        raise HTTPException(503, 'Admin access is not configured. Set ADMIN_API_KEY before deployment.')
    if not required:
        return
    header_ok = bool(x_admin_key) and hmac.compare_digest(x_admin_key, required)
    if header_ok or _session_active(request.cookies.get(ADMIN_COOKIE)):
        return
    raise HTTPException(403, 'Admin key is missing or incorrect')


def enforce_chat_limit(request: Request):
    ip = request.client.host if request.client else 'local'
    now = time.monotonic()
    bucket = _rate_hits[ip]
    while bucket and now - bucket[0] > CHAT_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= CHAT_LIMIT:
        raise HTTPException(429, 'Too many questions. Wait a minute and try again.')
    bucket.append(now)


def enqueue(work) -> str:
    job_id = uuid.uuid4().hex[:16]
    with _state_lock:
        _jobs[job_id] = {'id': job_id, 'status': 'running', 'result': None, 'error': None}
        if len(_jobs) > 40:
            finished = [key for key, job in _jobs.items() if job['status'] != 'running']
            for key in finished[:len(_jobs) - 40]:
                _jobs.pop(key, None)

    def run():
        try:
            with _job_lock:
                result = work()
            with _state_lock:
                _jobs[job_id].update(status='done', result=result)
        except Exception as exc:
            message = str(exc) if isinstance(exc, (ValueError, OpenRouterError)) else 'The indexing job failed'
            with _state_lock:
                _jobs[job_id].update(status='error', error=message[:400])

    threading.Thread(target=run, daemon=True).start()
    return job_id


class HistoryTurn(BaseModel):
    question: str = Field(default='', max_length=1000)
    answer: str = Field(default='', max_length=2000)


class Question(BaseModel):
    question: str = Field(min_length=0, max_length=1500)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=4)
    stream: bool = False
    document_id: str = Field(default='', max_length=40)
    section: str = Field(default='', max_length=40)


class AdminLogin(BaseModel):
    key: str = Field(default='', max_length=200)

class EvalRequest(BaseModel):
    cases: list[dict] = Field(min_length=1, max_length=100)
    include_generation: bool = False
    judge: bool = False

class SweepRequest(BaseModel):
    cases: list[dict] = Field(min_length=1, max_length=30)
    top_k: list[int] = Field(default_factory=lambda: [3, 5, 8], max_length=5)
    min_similarity: list[float] = Field(default_factory=lambda: [0.30, 0.40, 0.55], max_length=5)

class PromptCompareRequest(BaseModel):
    cases: list[dict] = Field(min_length=1, max_length=15)
    prompts: list[str] = Field(default_factory=lambda: ['strict', 'concise', 'baseline'], max_length=3)

class BootstrapRequest(BaseModel):
    guideline_id: str
    rights_confirmed: bool = False
    background: bool = False

class FeedbackRequest(BaseModel):
    vote: str = Field(pattern='^(helpful|not_helpful)$')


def handle_error(exc: Exception):
    if isinstance(exc, ValueError):
        raise HTTPException(400, str(exc)) from exc
    if isinstance(exc, OpenRouterError):
        raise HTTPException(503, str(exc)) from exc
    raise exc


@app.get('/', include_in_schema=False)
def index():
    if not (WEB / 'index.html').exists():
        raise HTTPException(503, 'Web frontend missing; open /docs for API.')
    return FileResponse(WEB / 'index.html')


@app.get('/favicon.svg', include_in_schema=False)
def favicon():
    return FileResponse(WEB / 'favicon.svg', media_type='image/svg+xml')


@app.get('/health')
def health():
    rag = get_rag()
    docs = rag.store.list_documents()
    return {'status': 'ok', 'documents': len(docs), 'chunks': sum(d['chunk_count'] for d in docs),
            'provider': 'OpenRouter', 'configured': bool(rag.settings.api_key and rag.settings.api_key != 'replace_with_your_key'),
            'ready': bool(docs and rag.settings.api_key and rag.settings.api_key != 'replace_with_your_key'),
            'medical_use': 'educational_reference_only', 'version': '2.0.0'}


@app.get('/api/system')
def system_info():
    rag = get_rag()
    return {'chat_model': rag.settings.chat_model, 'embedding_model': rag.settings.embed_model,
            'top_k': rag.settings.top_k, 'min_similarity': rag.settings.min_similarity,
            'admin_configured': bool(rag.settings.admin_api_key and not rag.settings.admin_api_key.startswith('replace_')), 'production_mode': os.getenv('APP_ENV','').lower() == 'production'}


@app.get('/documents')
def documents():
    return get_rag().store.list_documents()


@app.get('/api/catalog')
def catalog():
    docs = get_rag().store.list_documents()
    urls = {x['source_url'] for x in docs}
    return [{**g.public(), 'indexed': g.pdf_url in urls} for g in CATALOG.values()]


@app.post('/api/admin/session')
def admin_login(body: AdminLogin, request: Request, response: Response):
    required = _admin_required()
    if required.startswith('replace_') or (not required and os.getenv('APP_ENV', '').lower() == 'production'):
        raise HTTPException(503, 'Set a unique ADMIN_API_KEY in .env before signing in')
    if not required:
        return {'required': False, 'active': True}
    if not body.key or not hmac.compare_digest(body.key, required):
        raise HTTPException(403, 'Admin key is missing or incorrect')
    token = secrets.token_urlsafe(32)
    with _state_lock:
        _sessions[token] = time.time() + 12 * 60 * 60
    response.set_cookie(
        ADMIN_COOKIE, token, httponly=True, samesite='strict', max_age=12 * 60 * 60, path='/',
        secure=request.url.scheme == 'https',
    )
    return {'required': True, 'active': True}


@app.delete('/api/admin/session')
def admin_logout(response: Response):
    response.delete_cookie(ADMIN_COOKIE, path='/')
    return {'active': False}


@app.get('/api/admin/session')
def admin_session(request: Request):
    required = _admin_required()
    if not required or required.startswith('replace_'):
        return {'required': False, 'active': not required.startswith('replace_')}
    return {'required': True, 'active': _session_active(request.cookies.get(ADMIN_COOKIE))}


@app.get('/api/jobs/{job_id}', dependencies=[Depends(protect_write)])
def job_status(job_id: str):
    with _state_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(404, 'Job not found')
        return dict(job)


@app.post('/documents/upload', dependencies=[Depends(protect_write)])
async def upload(file: UploadFile = File(...), title: str = Form(''), source_url: str = Form(''), rights_confirmed: bool = Form(False), background: bool = Form(False)):
    rag = get_rag()
    if os.getenv('APP_ENV', '').lower() == 'production' and not rights_confirmed:
        raise HTTPException(400, 'You must confirm lawful reuse rights and that the PDF contains no identifiable patient data')
    if source_url:
        parsed = urlsplit(source_url)
        if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password:
            raise HTTPException(400, 'Source URL must be a valid HTTPS URL')
    contents = await file.read(rag.settings.max_upload_mb * 1024 * 1024 + 1)
    filename = file.filename or 'uploaded.pdf'
    if background:
        job_id = enqueue(lambda data=contents, name=filename, doc_title=title, url=source_url: get_rag().ingest(data, name, title=doc_title, source_url=url))
        return {'job_id': job_id, 'status': 'running'}
    try:
        return rag.ingest(contents, filename, title=title, source_url=source_url)
    except Exception as exc:
        handle_error(exc)


def download_pdf(url: str, max_bytes: int) -> bytes:
    # Called ONLY with publisher URLs from the fixed allowlist, never a user URL.
    try:
        with httpx.Client(timeout=httpx.Timeout(110, connect=15), follow_redirects=True) as client:
            with client.stream('GET', url, headers={'User-Agent': 'ClinicalGuidelineIntelligence/2.0'}) as res:
                res.raise_for_status()
                payload = bytearray()
                for part in res.iter_bytes(128 * 1024):
                    payload.extend(part)
                    if len(payload) > max_bytes:
                        raise ValueError('Publisher PDF exceeds upload size limit')
                if not payload.startswith(b'%PDF'):
                    raise ValueError('Publisher did not return a valid PDF; use manual upload')
                return bytes(payload)
    except httpx.HTTPError as exc:
        raise ValueError(f'Publisher download unavailable ({type(exc).__name__}). Try again later or upload an authorised copy.') from exc


@app.post('/api/catalog/import', dependencies=[Depends(protect_write)])
def import_guideline(request: BootstrapRequest):
    guid = CATALOG.get(request.guideline_id)
    if guid is None:
        raise HTTPException(404, 'Not a supported publisher guideline')
    if not request.rights_confirmed:
        raise HTTPException(400, 'Review the source licence and confirm you have the necessary rights before import')
    rag = get_rag()
    if not rag.settings.api_key or rag.settings.api_key == 'replace_with_your_key':
        raise HTTPException(503, 'OpenRouter API key not configured in backend .env')
    label = f'{guid.publisher} — {guid.title} ({guid.year})'

    def work():
        content = download_pdf(guid.pdf_url, get_rag().settings.max_upload_mb * 1024 * 1024)
        return get_rag().ingest(content, f'{guid.id}.pdf', title=label, source_url=guid.pdf_url)

    if request.background:
        return {'job_id': enqueue(work), 'status': 'running'}
    try:
        return work()
    except Exception as exc:
        handle_error(exc)


@app.get('/documents/{document_id}/pdf')
def source_pdf(document_id: str):
    # Serve only indexed document IDs, and never accept a client-supplied path.
    indexed = {d['id'] for d in get_rag().store.list_documents()}
    if document_id not in indexed:
        raise HTTPException(404, 'Indexed PDF not found')
    path = Path(get_rag().settings.pdf_dir) / f'{document_id}.pdf'
    if not path.is_file():
        raise HTTPException(404, 'Original PDF not stored for this document')
    return FileResponse(path, media_type='application/pdf', headers={'X-Content-Type-Options': 'nosniff'})


@app.delete('/documents/{document_id}', dependencies=[Depends(protect_write)])
def remove_document(document_id: str):
    rag = get_rag()
    if not rag.store.delete_document(document_id):
        raise HTTPException(404, 'Document not found')
    (Path(rag.settings.pdf_dir) / f'{document_id}.pdf').unlink(missing_ok=True)
    return {'deleted': True, 'id': document_id}


@app.post('/retrieve')
def retrieve(request: Question, http_request: Request):
    enforce_chat_limit(http_request)
    try:
        history = [turn.model_dump() for turn in request.history]
        return {'results': get_rag().retrieve(request.question, history=history, document_id=request.document_id, section=request.section)}
    except Exception as exc:
        handle_error(exc)


def _sse(event: str, payload: dict) -> str:
    return f'event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n'


@app.post('/chat')
def chat(request: Question, http_request: Request):
    enforce_chat_limit(http_request)
    history = [turn.model_dump() for turn in request.history]
    if not request.stream:
        try:
            return get_rag().chat(request.question, history=history, document_id=request.document_id, section=request.section)
        except Exception as exc:
            handle_error(exc)

    def events():
        try:
            for event, payload in get_rag().iter_chat(request.question, history, document_id=request.document_id, section=request.section):
                yield _sse(event, payload)
        except Exception as exc:
            if isinstance(exc, ValueError):
                message = str(exc)
            elif isinstance(exc, OpenRouterError):
                message = str(exc)
            else:
                message = 'The answer could not be completed'
            yield _sse('error', {'message': message})

    return StreamingResponse(events(), media_type='text/event-stream')


@app.post('/api/feedback')
def feedback(request: FeedbackRequest):
    get_rag().store.record_feedback(request.vote)
    return {'saved': True}


@app.get('/api/feedback/stats', dependencies=[Depends(protect_write)])
def feedback_stats():
    return get_rag().store.feedback_stats()


@app.get('/api/evaluation/cases')
def evaluation_cases():
    return json.loads((ROOT / 'data' / 'eval_nice.json').read_text(encoding='utf-8'))


@app.post('/evaluation/run', dependencies=[Depends(protect_write)])
def evaluation(request: EvalRequest):
    try:
        return run_evaluation(get_rag(), request.cases, include_generation=request.include_generation, judge=request.judge)
    except Exception as exc:
        handle_error(exc)


@app.post('/evaluation/sweep', dependencies=[Depends(protect_write)])
def evaluation_sweep(request: SweepRequest):
    try:
        return sweep_retrieval(get_rag(), request.cases, request.top_k, request.min_similarity)
    except Exception as exc:
        handle_error(exc)


@app.post('/evaluation/prompts', dependencies=[Depends(protect_write)])
def evaluation_prompts(request: PromptCompareRequest):
    try:
        return compare_prompts(get_rag(), request.cases, request.prompts)
    except Exception as exc:
        handle_error(exc)
