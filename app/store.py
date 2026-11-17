"""Persistent metadata + normalized vectors + SQLite FTS5 lexical retrieval."""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
import re
from pathlib import Path
from typing import Any
import numpy as np

STOPWORDS = {'a', 'an', 'the', 'of', 'for', 'and', 'with', 'about', 'what', 'when', 'how', 'are', 'does', 'should', 'can', 'in', 'to', 'is', 'it', 'on', 'at', 'by', 'me', 'do', 'we', 'from', 'according', 'guideline', 'guidelines', 'who', 'which', 'where', 'this', 'that', 'those', 'these', 'they', 'them', 'also', 'same'}
# Short clinical abbreviations expanded only for the rerank overlap, not stored as new text.
QUERY_EXPANSIONS = {
    'bp': ('blood', 'pressure'),
    'abpm': ('ambulatory', 'monitoring'),
    'hbpm': ('home', 'monitoring'),
    'htn': ('hypertension',),
    'sbp': ('systolic',),
    'dbp': ('diastolic',),
}


def normalize(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if not np.all(np.isfinite(arr)) or np.any(norms <= 0):
        raise ValueError('Embedding returned zero or invalid vector')
    return arr / norms


class KnowledgeStore:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS configuration (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, vote TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL UNIQUE, source_url TEXT NOT NULL,
                    pages INTEGER NOT NULL, chunk_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
                    page INTEGER NOT NULL, section TEXT NOT NULL, recommendation TEXT,
                    text TEXT NOT NULL, vector BLOB NOT NULL, dimensions INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(id UNINDEXED, text, tokenize='porter unicode61');
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA busy_timeout=30000')
        try:
            with db:
                yield db
        finally:
            db.close()

    def list_documents(self) -> list[dict]:
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT * FROM documents ORDER BY indexed_at DESC, title')]

    def get_by_sha(self, sha: str) -> dict | None:
        with self.connect() as db:
            row = db.execute('SELECT * FROM documents WHERE sha256=?', (sha,)).fetchone()
            return dict(row) if row else None

    def check_embedding_model(self, model: str) -> None:
        with self.connect() as db:
            row = db.execute("SELECT value FROM configuration WHERE key='embedding_model'").fetchone()
            if row and row['value'] != model:
                raise ValueError(f'Embedding model mismatch: index uses {row["value"]}. Use the original model or reset the index.')

    def add_document(self, document: dict, chunks: list, vectors: np.ndarray, model: str) -> dict:
        if len(chunks) != len(vectors):
            raise ValueError('Embedding count does not match document chunk count')
        vectors = normalize(vectors)
        with self.connect() as db:
            existing = db.execute('SELECT value FROM configuration WHERE key=?', ('embedding_model',)).fetchone()
            if existing and existing['value'] != model:
                raise ValueError('Embedding model must match existing index')
            db.execute('INSERT OR IGNORE INTO configuration(key,value) VALUES (?,?)', ('embedding_model', model))
            db.execute('''INSERT INTO documents(id,title,filename,sha256,source_url,pages,chunk_count)
                          VALUES (:id,:title,:filename,:sha256,:source_url,:pages,:chunk_count)''', document)
            for chunk, vector in zip(chunks, vectors):
                db.execute('''INSERT INTO chunks(id,document_id,page,section,recommendation,text,vector,dimensions)
                              VALUES (?,?,?,?,?,?,?,?)''', (
                    f'{document["id"]}_{chunk.chunk_id}', document['id'], chunk.page,
                    chunk.section, chunk.recommendation, chunk.text,
                    vector.astype('<f4').tobytes(), vector.size))
                db.execute('INSERT INTO chunk_fts(id,text) VALUES (?,?)', (f'{document["id"]}_{chunk.chunk_id}', chunk.text))
        return document

    def delete_document(self, document_id: str) -> bool:
        with self.connect() as db:
            ids = [r['id'] for r in db.execute('SELECT id FROM chunks WHERE document_id=?', (document_id,))]
            for chunk_id in ids:
                db.execute('DELETE FROM chunk_fts WHERE id=?', (chunk_id,))
            db.execute('DELETE FROM chunks WHERE document_id=?', (document_id,))
            count = db.execute('DELETE FROM documents WHERE id=?', (document_id,)).rowcount
            if not db.execute('SELECT 1 FROM documents LIMIT 1').fetchone():
                db.execute("DELETE FROM configuration WHERE key='embedding_model'")
            return count > 0

    def search(self, query: str, query_vector: np.ndarray, *, top_k: int, min_similarity: float, document_id: str | None = None, section: str | None = None) -> list[dict[str, Any]]:
        query_vector = normalize(query_vector)[0]
        with self.connect() as db:
            rows = [dict(r) for r in db.execute('''
                SELECT c.*, d.title, d.filename, d.source_url
                FROM chunks c JOIN documents d ON d.id = c.document_id''')]
            if not rows:
                return []
            dim = rows[0]['dimensions']
            if query_vector.size != dim:
                raise ValueError('Query embedding dimensions differ from index')
            embeddings = np.vstack([np.frombuffer(r['vector'], dtype='<f4') for r in rows])
            similarity = embeddings @ query_vector
            wanted_section = (section or '').strip().lower()
            candidates = []
            for row, score in zip(rows, similarity):
                if document_id and row['document_id'] != document_id:
                    continue
                if wanted_section and not str(row.get('section', '')).lower().startswith(wanted_section):
                    continue
                if float(score) >= min_similarity:
                    row.pop('vector', None)
                    row.pop('dimensions', None)
                    row['similarity'] = round(float(score), 4)
                    candidates.append(row)
            if not candidates:
                return []
            # Semantic cosine ranking + SQLite BM25 lexical ranking via reciprocal rank fusion.
            ranked = sorted(candidates, key=lambda x: x['similarity'], reverse=True)
            semantic_rank = {r['id']: i for i, r in enumerate(ranked, 1)}
            tokens = [x for x in re.findall(r'[a-z0-9]+', query.lower()) if x not in STOPWORDS and len(x) >= 3]
            tokens = list(dict.fromkeys(tokens))[:18]
            lexical_rank: dict[str, int] = {}
            if tokens:
                match_query = ' OR '.join('"' + t + '"' for t in tokens)
                try:
                    for i, r in enumerate(db.execute('SELECT id FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY bm25(chunk_fts) LIMIT 200', (match_query,)), 1):
                        lexical_rank[r['id']] = i
                except sqlite3.OperationalError:
                    pass
            terms, numbers = _rerank_features(query)
            for row in candidates:
                sid = row['id']
                row['hybrid_score'] = round(1 / (60 + semantic_rank[sid]) + (1 / (60 + lexical_rank[sid]) if sid in lexical_rank else 0), 6)
                term_score, number_score = _passage_overlap(row.get('text', ''), row.get('section', ''), terms, numbers)
                # Second-stage rerank: keep cosine dominant, then lexical coverage, then exact numbers.
                row['rerank_score'] = round(0.62 * row['similarity'] + 0.23 * term_score + 0.15 * number_score, 4)
            candidates.sort(key=lambda r: (r['rerank_score'], r['similarity'], r['hybrid_score']), reverse=True)
            return candidates[:top_k]

    def record_feedback(self, vote: str) -> None:
        if vote not in ('helpful', 'not_helpful'):
            raise ValueError('Unknown feedback vote')
        with self.connect() as db:
            db.execute('INSERT INTO feedback(vote) VALUES (?)', (vote,))

    def feedback_stats(self) -> dict:
        with self.connect() as db:
            rows = db.execute('SELECT vote, COUNT(*) AS count FROM feedback GROUP BY vote')
            return {r['vote']: r['count'] for r in rows}


def _rerank_features(query: str) -> tuple[list[str], list[str]]:
    raw = re.findall(r'[a-z0-9]+', query.lower())
    terms: list[str] = []
    for token in raw:
        if token in STOPWORDS or len(token) < 3 or token.isdigit():
            continue
        terms.append(token)
        terms.extend(QUERY_EXPANSIONS.get(token, ()))
    numbers = [re.sub(r'\s+', '', n) for n in re.findall(r'\d+(?:\s*/\s*\d+)?', query)]
    return list(dict.fromkeys(terms))[:24], list(dict.fromkeys(numbers))[:8]


def _passage_overlap(text: str, section: str, terms: list[str], numbers: list[str]) -> tuple[float, float]:
    hay = f'{section}\n{text}'.lower()
    compact = re.sub(r'\s+', '', hay)
    if terms:
        hits = 0
        for term in terms:
            stem = term[:5] if len(term) >= 5 else term
            if term in hay or (len(stem) >= 4 and stem in hay):
                hits += 1
        term_score = hits / len(terms)
    else:
        term_score = 0.0
    if numbers:
        number_score = sum(1 for number in numbers if number in compact) / len(numbers)
    else:
        number_score = 0.0
    return term_score, number_score
