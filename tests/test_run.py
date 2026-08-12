"""End-to-end proof of the P1 acceptance criterion (SPEC.md §18):

    "evals/run.py can score a dummy retrieval function end to end and
    print all metrics with CIs."

Uses a dummy retrieval function (trivial keyword overlap, not a real
pattern) and MockLLM wired in as BOTH the generation and judge backend, so
this test proves the full run_pattern() orchestration -- retrieval metrics
AND judge-derived metrics -- with zero real API calls (SPEC.md R8).
"""

import json

from recipes import AnswerWithCitations
from recipes.llm import MockLLM
from evals.run import run_pattern

FIXTURE_CORPUS = {
    "c1": {
        "chunk_id": "c1",
        "text": "Speculative decoding uses a small draft model to propose "
        "tokens that a larger model verifies in parallel, reducing latency.",
    },
    "c2": {
        "chunk_id": "c2",
        "text": "Convolutional neural networks are widely used for image "
        "classification tasks.",
    },
    "c3": {
        "chunk_id": "c3",
        "text": "Reinforcement learning from human feedback aligns language "
        "model outputs with human preferences.",
    },
}

FIXTURE_QA_SET = [
    {
        "qid": "q1",
        "category": "keyword",
        "question": "What is speculative decoding?",
        "relevant_chunk_ids": ["c1"],
        "reference_answer": "A technique using a draft model to speed up inference.",
        "requires_filter": None,
    },
    {
        "qid": "q2",
        "category": "keyword",
        "question": "What are CNNs used for?",
        "relevant_chunk_ids": ["c2"],
        "reference_answer": "Image classification.",
        "requires_filter": None,
    },
]


def _dummy_pattern_factory():
    """A trivial retrieval function: ranks fixture chunks by naive keyword
    overlap with the question. Not a real pattern -- exists only to drive
    run_pattern() end to end for this test.
    """
    gen_llm = MockLLM(default_response="This is a test answer. [c1]")

    def dummy_pattern(question: str, k: int = 5) -> AnswerWithCitations:
        q_words = set(question.lower().split())
        scored = []
        for chunk_id, chunk in FIXTURE_CORPUS.items():
            overlap = len(q_words & set(chunk["text"].lower().split()))
            scored.append((overlap, chunk_id))
        scored.sort(reverse=True)
        retrieved = [cid for _, cid in scored[:k]]

        response = gen_llm.complete(
            prompt=f"Answer using context: {question}",
            model="gpt-4.1-mini-2025-04-14",
        )
        return AnswerWithCitations(
            answer=response.text,
            retrieved_chunk_ids=retrieved,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_input_tokens=response.cached_input_tokens,
        )

    return dummy_pattern


def test_run_pattern_end_to_end_with_judges(capsys):
    dummy_pattern = _dummy_pattern_factory()
    judge_llm = MockLLM(
        default_response='{"score": 1, "reasoning": "looks fine"}'
    )

    result = run_pattern(
        recipe_fn=dummy_pattern,
        qa_set=FIXTURE_QA_SET,
        corpus_by_id=FIXTURE_CORPUS,
        llm=judge_llm,
        pattern_name="dummy",
        judges_enabled=True,
        k=3,
        verbose=True,
    )

    assert result.n_questions == 2

    # Retrieval metrics: q1 should hit (c1 is top by keyword overlap).
    assert result.hit_at_3.mean > 0.0
    assert result.hit_at_10.mean > 0.0
    assert result.mrr.mean > 0.0
    assert result.hit_at_3.lower <= result.hit_at_3.mean <= result.hit_at_3.upper

    # Judge-derived metrics: this is the part a retrieval-only test would
    # miss. MockLLM returns score=1 for every judge call, so these must be
    # populated and equal to 1.0, not None.
    assert result.faithfulness is not None
    assert result.faithfulness.mean == 1.0
    assert result.answer_relevance is not None
    assert result.answer_relevance.mean == 1.0
    assert result.citation_accuracy is not None
    assert result.citation_accuracy.mean == 1.0

    # No filter-dependent questions in this fixture set.
    assert result.filter_accuracy is None

    # Cost and latency are computed, not just present as zeros by accident.
    assert result.usd_per_query > 0.0
    assert result.eval_usd > 0.0
    assert result.p50_latency_ms >= 0.0

    # "print all metrics with CIs" -- verify the printed summary actually
    # names every metric, per the literal wording of the P1 acceptance
    # criterion.
    printed = capsys.readouterr().out
    for label in [
        "hit@3",
        "hit@10",
        "mrr",
        "faithfulness",
        "answer_relevance",
        "citation_accuracy",
        "95% CI",
    ]:
        assert label in printed


def test_run_pattern_judges_disabled_are_none():
    dummy_pattern = _dummy_pattern_factory()
    result = run_pattern(
        recipe_fn=dummy_pattern,
        qa_set=FIXTURE_QA_SET,
        corpus_by_id=FIXTURE_CORPUS,
        llm=MockLLM(),
        judges_enabled=False,
        verbose=False,
    )
    assert result.faithfulness is None
    assert result.answer_relevance is None
    assert result.citation_accuracy is None
    # Retrieval metrics still work without judges.
    assert result.hit_at_3.mean >= 0.0
    assert result.n_errors == 0


def test_run_pattern_isolates_a_failing_question(capsys):
    """Regression test for review fix #3: one recipe_fn failure must not
    lose the scores already collected for other questions in the run.
    """

    def flaky_pattern(question: str, k: int = 5):
        if question == FIXTURE_QA_SET[0]["question"]:
            raise RuntimeError("simulated recipe failure")
        return _dummy_pattern_factory()(question, k=k)

    result = run_pattern(
        recipe_fn=flaky_pattern,
        qa_set=FIXTURE_QA_SET,
        corpus_by_id=FIXTURE_CORPUS,
        llm=MockLLM(default_response='{"score": 1, "reasoning": "fine"}'),
        judges_enabled=True,
        verbose=False,
    )

    # One of the two fixture questions failed; the other's scores must
    # still be present, not discarded.
    assert result.n_errors == 1
    assert result.hit_at_3.mean >= 0.0
    assert result.faithfulness is not None

    printed = capsys.readouterr().err
    assert "failed and was skipped" in printed


def test_run_pattern_keeps_retrieval_scores_when_only_judging_fails(capsys):
    """Regression test for a bug found while building 02_bm25.ipynb (P2):
    a judge-call failure must NOT discard the retrieval scores for that
    same question -- retrieval already succeeded and is real, valid data.
    Previously a judge failure was folded into the same n_errors/"failed
    and was skipped" path as a recipe_fn failure, which was both wrong
    (real data thrown away) and misleading (the message implied nothing
    was captured for that question, when hit@k/mrr were).
    """
    dummy_pattern = _dummy_pattern_factory()
    # Not valid JSON -- every judge call will raise JudgeParseError.
    non_json_llm = MockLLM(default_response="This is a mock response.")

    result = run_pattern(
        recipe_fn=dummy_pattern,
        qa_set=FIXTURE_QA_SET,
        corpus_by_id=FIXTURE_CORPUS,
        llm=non_json_llm,
        judges_enabled=True,
        verbose=False,
    )

    # Recipe_fn never failed -- n_errors stays 0.
    assert result.n_errors == 0
    # Every question's judging failed -- n_judge_errors reflects that.
    assert result.n_judge_errors == len(FIXTURE_QA_SET)
    # Retrieval metrics are real and were NOT discarded by the judge failure.
    assert result.n_questions == len(FIXTURE_QA_SET)
    assert result.hit_at_3.mean > 0.0
    # Judge-derived metrics are empty (bootstrap_ci's empty-list default),
    # since every judge call failed to parse.
    assert result.faithfulness.mean == 0.0

    printed = capsys.readouterr().err
    assert "retrieval succeeded but judging failed" in printed
    assert "failed and was skipped (no retrieval data)" not in printed


def test_run_pattern_keeps_retrieval_scores_when_only_accounting_fails(capsys):
    """Regression test for a bug found in code review: cost_usd() and
    filter_accuracy() ran in the SAME try block as the hit@k/mrr/latency
    appends, so a cost-accounting failure (e.g. an unpriced generation
    model) would print "no retrieval data" even though retrieval scores
    for that question were already appended and kept. Same bug shape as
    the judge-error split above, applied to the accounting step.
    """
    dummy_pattern = _dummy_pattern_factory()

    result = run_pattern(
        recipe_fn=dummy_pattern,
        qa_set=FIXTURE_QA_SET,
        corpus_by_id=FIXTURE_CORPUS,
        llm=MockLLM(default_response='{"score": 1, "reasoning": "fine"}'),
        # Not a real priced model -- cost_usd() raises ValueError for this,
        # deliberately, to exercise the accounting-error path. recipe_fn's
        # own internal llm.complete() call is unaffected (MockLLM doesn't
        # care about pricing), so retrieval itself still succeeds.
        generation_model="totally-unpriced-model-xyz",
        judges_enabled=True,
        verbose=False,
    )

    # Recipe_fn never failed -- n_errors stays 0.
    assert result.n_errors == 0
    # Cost accounting failed for every question.
    assert result.n_accounting_errors == len(FIXTURE_QA_SET)
    # Retrieval metrics are real and were NOT discarded by the accounting failure.
    assert result.n_questions == len(FIXTURE_QA_SET)
    assert result.hit_at_3.mean > 0.0
    # Judging still ran fine (independent of accounting) since it doesn't
    # depend on cost_usd() succeeding -- and judge cost IS counted in
    # eval_usd (judges have their own cost_usd() call, unaffected by this
    # question's generation-cost failure), so eval_usd is small but not
    # exactly zero. What's understated is specifically the generation-cost
    # contribution, not the total.
    assert result.faithfulness.mean == 1.0
    assert result.eval_usd > 0.0  # judge cost alone
    assert result.eval_usd < 0.01  # sanity: nowhere near what generation cost would add

    printed = capsys.readouterr().err
    assert "retrieval succeeded but cost/filter accounting failed" in printed
    assert "failed and was skipped (no retrieval data)" not in printed


def test_run_pattern_writes_tool_call_traces_when_path_given(tmp_path):
    """Regression test for P4 (pattern 10): trace_output_path, when given,
    must get one JSON line per question with a non-empty tool_call_trace.
    """
    gen_llm = MockLLM(default_response="answer")

    def traced_pattern(question: str, k: int = 5) -> AnswerWithCitations:
        response = gen_llm.complete(prompt=question, model="gpt-4.1-mini-2025-04-14")
        return AnswerWithCitations(
            answer=response.text,
            retrieved_chunk_ids=["c1"],
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            tool_call_trace=[{"action": "search_dense", "action_input": question}],
        )

    trace_path = tmp_path / "agentic_traces.jsonl"
    run_pattern(
        recipe_fn=traced_pattern,
        qa_set=FIXTURE_QA_SET,
        corpus_by_id=FIXTURE_CORPUS,
        llm=MockLLM(),
        judges_enabled=False,
        verbose=False,
        trace_output_path=trace_path,
    )

    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(FIXTURE_QA_SET)
    for line, record in zip(lines, FIXTURE_QA_SET):
        parsed = json.loads(line)
        assert parsed["qid"] == record["qid"]
        assert parsed["question"] == record["question"]
        assert parsed["trace"] == [{"action": "search_dense", "action_input": record["question"]}]


def test_run_pattern_without_trace_output_path_is_unaffected():
    # Omitting trace_output_path (the existing, default call shape) must
    # still work exactly as before -- proves the new param has zero impact
    # when unused.
    dummy_pattern = _dummy_pattern_factory()
    result = run_pattern(
        recipe_fn=dummy_pattern,
        qa_set=FIXTURE_QA_SET,
        corpus_by_id=FIXTURE_CORPUS,
        llm=MockLLM(),
        judges_enabled=False,
        verbose=False,
    )
    assert result.n_questions == len(FIXTURE_QA_SET)
