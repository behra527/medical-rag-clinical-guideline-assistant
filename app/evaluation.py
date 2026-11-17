"""Transparent reproducible retrieval and answer-evidence evaluation."""
import json
from statistics import mean
from time import perf_counter
from pathlib import Path


def judge_answer(rag, question: str, answer: str, sources: list[dict], expected_terms: list[str]) -> dict | None:
    """Optional LLM-as-judge; never a substitute for independent clinician grading."""
    evidence = "\n".join(f"[{s['id']}] {s['excerpt']}" for s in sources)
    instruction = (
        "You are an independent evaluator for a clinical guideline QA experiment. "
        "Use only the supplied evidence; do not assume claims are true without support. "
        "Assess correctness, groundedness, completeness and safety, each integer 1 through 5. "
        "Return ONLY JSON with keys correctness, groundedness, completeness, safety, rationale. "
        "Uncited or unsupported medical claims must lower the scores. "
        "This is a noisy automated judgement, not clinical validation."
    )
    payload = f"Question: {question}\nAnswer: {answer}\nExpected concepts: {expected_terms}\nEvidence: {evidence}"
    try:
        raw = rag.client.complete([{'role': 'system', 'content': instruction},
                                   {'role': 'user', 'content': payload}], max_tokens=350)
        raw = raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
        scores = json.loads(raw)
        keys = ('correctness', 'groundedness', 'completeness', 'safety')
        if not all(isinstance(scores.get(k), int) and 1 <= scores[k] <= 5 for k in keys):
            return None
        return {**{k: scores[k] for k in keys}, 'rationale': str(scores.get('rationale', ''))[:600]}
    except (ValueError, TypeError, KeyError, RuntimeError):
        return None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * pct
    a = int(idx)
    b = min(a + 1, len(s) - 1)
    return round(s[a] + (s[b] - s[a]) * (idx - a), 1)


def run_evaluation(rag, cases: list[dict], include_generation: bool = True, judge: bool = False) -> dict:
    if not cases:
        raise ValueError('Empty evaluation set')
    result_rows = []
    for case in cases:
        q = case['question']
        expected_abstain = bool(case.get('should_abstain', False))
        start = perf_counter()
        retrieved = rag.retrieve(q)
        retrieval_ms = (perf_counter() - start) * 1000
        expected_section = case.get('gold_section', '')
        expected_doc = case.get('gold_document', '')
        ranks = [i for i, row in enumerate(retrieved, start=1)
                 if (not expected_doc or expected_doc.lower() in row['title'].lower())
                 and (not expected_section or row['section'].startswith(expected_section))]
        rank = min(ranks) if ranks else None
        response = None
        evidence_fact_coverage = None
        if include_generation:
            response = rag.chat(q)
            if not response['abstained'] and case.get('expected_terms'):
                normalized = response['answer'].lower()
                evidence_fact_coverage = round(sum(term.lower() in normalized for term in case['expected_terms']) / len(case['expected_terms']), 3)
        judge_scores = None
        if judge and response and not response['abstained']:
            judge_scores = judge_answer(rag, q, response['answer'], response['sources'], case.get('expected_terms', []))
        result_rows.append({
            'question': q, 'retrieval_hit_at_k': bool(rank) if not expected_abstain else not retrieved,
            'reciprocal_rank': round(1 / rank, 4) if rank else 0,
            'first_relevant_rank': rank, 'retrieval_time_ms': round(retrieval_ms, 1),
            'answer_time_ms': response['response_time_ms'] if response else None,
            'abstained': response['abstained'] if response else None,
            'correct_abstention': (response['abstained'] == expected_abstain) if response else None,
            'citation_valid': bool(response['sources']) if response and not response['abstained'] else None,
            'expected_term_coverage': evidence_fact_coverage,
            'judge_scores': judge_scores,
            'answer': response['answer'] if response else None,
            'sources': response['sources'] if response else [],
        })
    positives = [r for r, c in zip(result_rows, cases) if not c.get('should_abstain')]
    answers = [r for r in result_rows if r['answer_time_ms'] is not None]
    fact_coverage = [r['expected_term_coverage'] for r in result_rows if r['expected_term_coverage'] is not None]
    graded = [r['judge_scores'] for r in result_rows if r['judge_scores']]
    return {'n_cases': len(cases), 'n_positive_cases': len(positives),
            'metrics': {
                'retrieval_hit_at_k': round(mean(r['retrieval_hit_at_k'] for r in positives), 3) if positives else None,
                'mrr_at_k': round(mean(r['reciprocal_rank'] for r in positives), 3) if positives else None,
                'mean_retrieval_ms': round(mean(r['retrieval_time_ms'] for r in result_rows), 1),
                'mean_answer_ms': round(mean(r['answer_time_ms'] for r in answers), 1) if answers else None,
                'p50_answer_ms': _percentile([r['answer_time_ms'] for r in answers], .5) if answers else None,
                'p95_answer_ms': _percentile([r['answer_time_ms'] for r in answers], .95) if answers else None,
                'abstention_accuracy': round(mean(r['correct_abstention'] for r in answers), 3) if answers else None,
                'citation_presence_rate': round(mean(r['citation_valid'] for r in answers if r['citation_valid'] is not None), 3) if any(r['citation_valid'] is not None for r in answers) else None,
                'expected_term_coverage': round(mean(fact_coverage), 3) if fact_coverage else None,
                'llm_judge_mean_correctness_1_to_5': round(mean(r['correctness'] for r in graded), 2) if graded else None,
                'llm_judge_mean_groundedness_1_to_5': round(mean(r['groundedness'] for r in graded), 2) if graded else None,
            },
            'metric_notes': {
                'retrieval_hit_at_k': 'Gold source document + section appears in the retrieved top-k (not clinical correctness).',
                'mrr_at_k': 'Mean reciprocal rank of first correct source section, over positive questions.',
                'expected_term_coverage': 'Weak lexical answer-quality proxy. Requires expert review for medical validity.',
                'citation_presence_rate': 'Checks evidence IDs are present and valid, NOT whether every claim is entailed.',
                'response_time': 'Wall clock; dependent on network, provider and document count.',
                'llm_judge': 'Optional automated ratings 1-5; may be biased/inconsistent. Not a clinician assessment.',
            }, 'cases': result_rows}


def _passage_status(retrieved: list[dict], case: dict) -> str:
    expected_abstain = bool(case.get('should_abstain', False))
    if expected_abstain:
        return 'irrelevant' if retrieved else 'correct_abstain'
    if not retrieved:
        return 'insufficient'
    expected_section = case.get('gold_section', '')
    expected_doc = case.get('gold_document', '')
    hit = any((not expected_doc or expected_doc.lower() in row['title'].lower()) and (not expected_section or str(row['section']).startswith(expected_section)) for row in retrieved)
    return 'relevant' if hit else 'irrelevant'


def sweep_retrieval(rag, cases: list[dict], top_ks: list[int], thresholds: list[float]) -> dict:
    """Reuse one embedding per question across the top-k and threshold grid."""
    if not cases:
        raise ValueError('Empty evaluation set')
    if not top_ks or not thresholds or len(top_ks) > 5 or len(thresholds) > 5:
        raise ValueError('Provide 1 to 5 top-k values and 1 to 5 thresholds')
    if any(not 1 <= int(k) <= 20 for k in top_ks) or any(not 0 <= float(t) <= 1 for t in thresholds):
        raise ValueError('top-k must be 1–20 and similarity must be within 0–1')
    from .rag import validate_question
    prepared = []
    has_docs = bool(rag.store.list_documents())
    if has_docs:
        rag.store.check_embedding_model(rag.settings.embed_model)
    for case in cases:
        question = validate_question(case['question'])
        embedding = rag.client.embed([question])[0] if has_docs else None
        prepared.append((case, question, embedding))
    cells = []
    focus = None
    for top_k in top_ks:
        for threshold in thresholds:
            counts = {'relevant': 0, 'irrelevant': 0, 'insufficient': 0, 'correct_abstain': 0}
            problems = []
            for case, question, embedding in prepared:
                retrieved = [] if embedding is None else rag.store.search(question, embedding, top_k=int(top_k), min_similarity=float(threshold))
                status = _passage_status(retrieved, case)
                counts[status] += 1
                if status in ('irrelevant', 'insufficient'):
                    top = retrieved[0] if retrieved else None
                    problems.append({
                        'question': question, 'status': status,
                        'gold_section': case.get('gold_section', ''),
                        'top_section': top['section'] if top else '',
                        'top_similarity': top['similarity'] if top else None,
                    })
            graded = counts['relevant'] + counts['irrelevant'] + counts['insufficient']
            cell = {
                'top_k': int(top_k), 'min_similarity': float(threshold), **counts,
                'relevant_rate': round(counts['relevant'] / graded, 3) if graded else None,
                'problems': problems,
            }
            cells.append(cell)
            if top_k == 5 and abs(float(threshold) - 0.4) < 1e-9:
                focus = cell
    if focus is None:
        focus = cells[0]
    return {
        'n_cases': len(cases), 'cells': [{k: v for k, v in cell.items() if k != 'problems'} | {'problem_count': len(cell['problems'])} for cell in cells],
        'problems': focus['problems'],
        'focus': {'top_k': focus['top_k'], 'min_similarity': focus['min_similarity']},
        'note': 'relevant = gold section is inside top-k. irrelevant = passages came back for the wrong section, or for a question that should abstain. insufficient = nothing passed the threshold.',
    }


def compare_prompts(rag, cases: list[dict], prompt_names: list[str]) -> dict:
    from .rag import PROMPTS
    if not cases:
        raise ValueError('Empty evaluation set')
    if not prompt_names or any(name not in PROMPTS for name in prompt_names):
        raise ValueError('Prompt must be strict, concise, or baseline')
    variants = []
    for name in prompt_names:
        rows = []
        for case in cases:
            response = rag.chat(case['question'], prompt=name)
            expected = [term.lower() for term in case.get('expected_terms', [])]
            coverage = None
            if expected and not response['abstained']:
                text = response['answer'].lower()
                coverage = round(sum(term in text for term in expected) / len(expected), 3)
            rows.append({
                'question': case['question'], 'abstained': response['abstained'],
                'unsupported_claims': len(response.get('unsupported_claims') or []),
                'expected_term_coverage': coverage,
                'citation_present': bool(response.get('sources')),
                'answer': response['answer'],
            })
        answered = [row for row in rows if row['expected_term_coverage'] is not None]
        variants.append({
            'prompt': name,
            'metrics': {
                'abstentions': sum(1 for row in rows if row['abstained']),
                'unsupported_claims': sum(row['unsupported_claims'] for row in rows),
                'expected_term_coverage': round(mean(row['expected_term_coverage'] for row in answered), 3) if answered else None,
                'citation_presence_rate': round(mean(row['citation_present'] for row in rows if not row['abstained']), 3) if any(not row['abstained'] for row in rows) else None,
            },
            'cases': rows,
        })
    return {'variants': variants, 'note': 'Same questions, different instructions. Unsupported claims are sentences removed because a number, acronym, or page was not in the cited passage.'}


def load_dataset(path: str):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, list) or not all(isinstance(c, dict) and 'question' in c for c in data):
        raise ValueError('Dataset must be a JSON array with question for each case')
    return data
