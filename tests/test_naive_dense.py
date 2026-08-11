from recipes.embeddings import MockEmbedder
from recipes.llm import MockLLM
from recipes.naive_dense import make_retrieve_and_answer

FIXTURE_CORPUS = {
    "c1": {"chunk_id": "c1", "text": "Speculative decoding speeds up LLM inference."},
    "c2": {"chunk_id": "c2", "text": "Convolutional networks are used for image classification."},
    "c3": {"chunk_id": "c3", "text": "Reinforcement learning aligns model outputs with feedback."},
}


def test_make_retrieve_and_answer_returns_bound_closure():
    fn = make_retrieve_and_answer(
        FIXTURE_CORPUS, embedder=MockEmbedder(), llm=MockLLM(default_response="answer text")
    )
    result = fn("What speeds up inference?", k=2)

    assert result.answer == "answer text"
    assert len(result.retrieved_chunk_ids) == 2
    assert all(cid in FIXTURE_CORPUS for cid in result.retrieved_chunk_ids)
    assert result.latency_ms >= 0.0
    assert result.input_tokens > 0
    assert result.output_tokens > 0


def test_retrieved_chunk_ids_come_from_the_real_corpus_keys():
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=MockLLM())
    result = fn("any question", k=10)
    # k=10 exceeds corpus size (3); should return all 3, not error or pad.
    assert set(result.retrieved_chunk_ids) == set(FIXTURE_CORPUS.keys())


def test_vector_store_dim_matches_embedder_not_hardcoded():
    # A MockEmbedder with a non-default dim must still work end to end --
    # proves the dim is read from the real embed() output, not assumed.
    fn = make_retrieve_and_answer(
        FIXTURE_CORPUS, embedder=MockEmbedder(dim=8), llm=MockLLM()
    )
    result = fn("test question", k=1)
    assert len(result.retrieved_chunk_ids) == 1


def test_context_passed_to_llm_contains_retrieved_chunk_text():
    llm = MockLLM()
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm)
    fn("What speeds up inference?", k=1)

    assert len(llm.calls) == 1
    prompt = llm.calls[0].prompt
    # The prompt must include at least one real chunk's text (not empty
    # context), proving retrieval results actually flow into generation.
    assert any(chunk["text"] in prompt for chunk in FIXTURE_CORPUS.values())
