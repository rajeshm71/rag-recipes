from recipes.embeddings import MockEmbedder
from recipes.hyde import make_retrieve_and_answer
from recipes.llm import MockLLM

FIXTURE_CORPUS = {
    "c1": {"chunk_id": "c1", "text": "Speculative decoding speeds up LLM inference."},
    "c2": {"chunk_id": "c2", "text": "Convolutional networks are used for image classification."},
    "c3": {"chunk_id": "c3", "text": "Reinforcement learning aligns model outputs with feedback."},
}


def test_llm_is_called_exactly_twice_per_question():
    llm = MockLLM(default_response="A hypothetical passage about the topic.")
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm)
    fn("What speeds up inference?", k=2)
    assert len(llm.calls) == 2


def test_first_call_has_no_context_second_call_does():
    llm = MockLLM(default_response="A hypothetical passage about the topic.")
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm)
    fn("What speeds up inference?", k=2)

    first_prompt = llm.calls[0].prompt
    second_prompt = llm.calls[1].prompt

    chunk_texts = [c["text"] for c in FIXTURE_CORPUS.values()]
    assert not any(text in first_prompt for text in chunk_texts)
    assert any(text in second_prompt for text in chunk_texts)


def test_result_answer_comes_from_the_second_call():
    llm = MockLLM(
        canned={
            "Passage:": "hypothetical text",
        },
        default_response="final answer text",
    )
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm)
    result = fn("What speeds up inference?", k=2)
    assert result.answer == "final answer text"


def test_token_counts_are_summed_across_both_calls():
    llm = MockLLM(default_response="A hypothetical passage about the topic.")
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm)
    result = fn("What speeds up inference?", k=2)

    assert len(llm.calls) == 2
    # Recompute what each individual mock call would have reported, using
    # the same deterministic pseudo-token formula MockLLM uses internally,
    # and assert the pattern's summed total matches -- proves neither call's
    # tokens are silently dropped.
    expected_input = sum(max(1, len(c.prompt) // 4) for c in llm.calls)
    assert result.input_tokens == expected_input
