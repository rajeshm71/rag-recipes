"""Tests for recipes/bm25.py's make_retrieve_and_answer() pattern factory.
Distinct from tests/test_bm25.py, which covers the underlying BM25Index
infrastructure class directly.
"""

from recipes.bm25 import make_retrieve_and_answer
from recipes.llm import MockLLM

FIXTURE_CORPUS = {
    "c1": {"chunk_id": "c1", "text": "Speculative decoding speeds up LLM inference significantly."},
    "c2": {"chunk_id": "c2", "text": "This paper discusses image classification with CNNs."},
    "c3": {"chunk_id": "c3", "text": "Reinforcement learning from human feedback aligns models."},
}


def test_bm25_pattern_ranks_exact_keyword_match_first():
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, llm=MockLLM(default_response="answer"))
    result = fn("speculative decoding inference", k=2)

    assert result.retrieved_chunk_ids[0] == "c1"
    assert result.answer == "answer"


def test_bm25_pattern_returns_bound_closure_shape():
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, llm=MockLLM())
    result = fn("decoding inference", k=1)

    assert len(result.retrieved_chunk_ids) == 1
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert result.latency_ms >= 0.0


def test_bm25_pattern_context_reaches_llm():
    llm = MockLLM()
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, llm=llm)
    fn("speculative decoding", k=1)

    assert "Speculative decoding" in llm.calls[0].prompt


def test_bm25_pattern_no_shared_keywords_returns_empty_not_arbitrary():
    # A query with zero shared tokens is BM25's defining failure mode: it
    # must surface as an empty retrieval (and an empty context for the
    # generator), not as k arbitrary chunks that happen to sort first.
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, llm=MockLLM())
    result = fn("zzz nonexistent term qqq", k=3)
    assert result.retrieved_chunk_ids == []
    assert result.answer  # generation still runs, just with no context
