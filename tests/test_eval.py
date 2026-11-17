from app.evaluation import run_evaluation


def test_retrieval_evaluation_and_answer_metrics(rag):
    cases = [
        {'question': 'How to diagnose blood pressure in adults?', 'gold_document': 'NICE NG136',
         'gold_section': '1.2', 'expected_terms': ['ABPM']},
        {'question': 'How do I fix my car engine?', 'should_abstain': True}
    ]
    report = run_evaluation(rag, cases)
    m = report['metrics']
    assert m['retrieval_hit_at_k'] == 1.0
    assert m['mrr_at_k'] == 1.0
    assert m['abstention_accuracy'] == 1.0
    assert m['expected_term_coverage'] == 1.0
    assert m['citation_presence_rate'] == 1.0
    assert len(report['cases']) == 2


def test_retrieval_sweep_flags_wrong_section(rag):
    from app.evaluation import sweep_retrieval
    report = sweep_retrieval(rag, [
        {'question': 'How to diagnose blood pressure in adults?', 'gold_section': '1.2'},
        {'question': 'How to diagnose blood pressure in adults?', 'gold_section': '1.4'},
        {'question': 'How do I fix my car engine?', 'should_abstain': True},
    ], [1], [0.5])
    cell = report['cells'][0]
    assert cell['relevant'] == 1
    assert cell['irrelevant'] == 1
    assert cell['correct_abstain'] == 1
    assert report['problems'][0]['status'] == 'irrelevant'


def test_prompt_comparison_catches_weaker_prompt(rag):
    from app.evaluation import compare_prompts
    from .conftest import FakeOpenRouter

    class Switching(FakeOpenRouter):
        def complete(self, messages, **kwargs):
            self.calls += 1
            text = messages[0]['content']
            if 'Citations are optional' in text:
                return 'ABPM is recommended [S1]. Also take 999 mg daily [S1].'
            if 'at most 60 words' in text:
                return 'ABPM is recommended [S1].'
            return 'Based on the guideline, ABPM is recommended [S1].'

    rag.client = Switching()
    report = compare_prompts(rag, [{'question': 'How do I diagnose blood pressure?', 'expected_terms': ['ABPM']}], ['strict', 'concise', 'baseline'])
    metrics = {item['prompt']: item['metrics'] for item in report['variants']}
    assert metrics['baseline']['unsupported_claims'] > metrics['concise']['unsupported_claims']
    assert metrics['concise']['expected_term_coverage'] == 1.0


def test_retrieval_only(rag):
    report = run_evaluation(rag, [{'question':'What blood pressure measurement?'}], include_generation=False)
    assert report['metrics']['mean_answer_ms'] is None
