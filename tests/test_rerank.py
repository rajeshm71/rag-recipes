from recipes.llm import MockLLM
from recipes.rerank import MockReranker, get_reranker, make_retrieve_and_answer

FIXTURE_CORPUS = {
    "c1": {"chunk_id": "c1", "text": "Speculative decoding speeds up LLM inference."},
    "c2": {"chunk_id": "c2", "text": "Convolutional networks are used for image classification."},
    "c3": {"chunk_id": "c3", "text": "Reinforcement learning aligns model outputs with feedback."},
}


def test_mock_reranker_is_deterministic():
    reranker = MockReranker()
    candidates = [(cid, c["text"]) for cid, c in FIXTURE_CORPUS.items()]
    order1 = [r.chunk_id for r in reranker.rerank("some query", candidates, top_k=3)]
    order2 = [r.chunk_id for r in reranker.rerank("some query", candidates, top_k=3)]
    assert order1 == order2


def test_mock_reranker_actually_reorders():
    # BM25's incoming order (insertion order) should not survive unchanged
    # for every query -- otherwise MockReranker is a no-op passthrough.
    reranker = MockReranker()
    candidates = [(cid, c["text"]) for cid, c in FIXTURE_CORPUS.items()]
    original_order = [cid for cid, _ in candidates]
    reordered = [r.chunk_id for r in reranker.rerank("some query", candidates, top_k=3)]
    assert reordered != original_order or len(candidates) == 1


def test_make_retrieve_and_answer_end_to_end_with_mock_reranker():
    fn = make_retrieve_and_answer(
        FIXTURE_CORPUS,
        llm=MockLLM(default_response="answer text"),
        reranker=MockReranker(),
    )
    result = fn("What speeds up inference?", k=2)

    assert result.answer == "answer text"
    assert len(result.retrieved_chunk_ids) == 2
    assert all(cid in FIXTURE_CORPUS for cid in result.retrieved_chunk_ids)


def test_get_reranker_respects_env_var(monkeypatch):
    monkeypatch.setenv("RAG_RECIPES_LLM", "mock")
    assert isinstance(get_reranker(), MockReranker)

    # Avoid actually constructing CrossEncoderReranker here -- its __init__
    # lazy-imports sentence_transformers and downloads a real model, which
    # unit tests must never do. Stub it out to prove get_reranker() picks
    # the real-backend class, without paying for a real model load.
    class _StubCrossEncoderReranker:
        pass

    monkeypatch.setattr("recipes.rerank.CrossEncoderReranker", _StubCrossEncoderReranker)
    monkeypatch.setenv("RAG_RECIPES_LLM", "openai")
    assert isinstance(get_reranker(), _StubCrossEncoderReranker)
