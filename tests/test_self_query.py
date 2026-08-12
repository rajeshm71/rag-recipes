from recipes.embeddings import MockEmbedder
from recipes.llm import MockLLM
from recipes.self_query import _parse_filter, make_retrieve_and_answer

FIXTURE_CORPUS = {
    "c1": {"chunk_id": "c1", "category": "cs.LG", "text": "Speculative decoding speeds up LLM inference."},
    "c2": {"chunk_id": "c2", "category": "cs.CV", "text": "Convolutional networks are used for image classification."},
    "c3": {"chunk_id": "c3", "category": "cs.LG", "text": "Reinforcement learning aligns model outputs with feedback."},
}

# Unique substring that only appears in self_query_prompt.txt, never in
# generation_prompt.txt -- lets MockLLM's `canned` dict distinguish calls.
FILTER_MARKER = "helping search a corpus of ML research papers"


def test_parse_filter_valid_json():
    assert _parse_filter('{"category": "cs.LG"}') == {"category": "cs.LG"}


def test_parse_filter_code_fenced_json():
    text = '```json\n{"category": "cs.CL"}\n```'
    assert _parse_filter(text) == {"category": "cs.CL"}


def test_parse_filter_garbage_falls_back_to_empty_dict():
    assert _parse_filter("not json at all") == {}


def test_parse_filter_non_dict_json_falls_back_to_empty_dict():
    assert _parse_filter('"just a string"') == {}


def test_parse_filter_empty_object():
    assert _parse_filter("{}") == {}


def test_extracted_filter_is_populated_on_result():
    llm = MockLLM(
        canned={FILTER_MARKER: '{"category": "cs.LG"}'}, default_response="final answer"
    )
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm)
    result = fn("Among the cs.LG papers, what speeds up inference?", k=5)
    assert result.extracted_filter == {"category": "cs.LG"}


def test_filter_actually_restricts_results():
    llm = MockLLM(
        canned={FILTER_MARKER: '{"category": "cs.LG"}'}, default_response="final answer"
    )
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm)
    result = fn("Among the cs.LG papers, what speeds up inference?", k=5)

    # Only c1 and c3 are cs.LG; c2 (cs.CV) must never appear.
    assert set(result.retrieved_chunk_ids).issubset({"c1", "c3"})
    assert "c2" not in result.retrieved_chunk_ids


def test_no_filter_question_searches_whole_corpus():
    llm = MockLLM(canned={FILTER_MARKER: "{}"}, default_response="final answer")
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm)
    result = fn("What speeds up inference?", k=5)
    assert result.extracted_filter == {}
    assert set(result.retrieved_chunk_ids) == set(FIXTURE_CORPUS.keys())


def test_filter_matching_zero_chunks_degrades_gracefully():
    llm = MockLLM(
        canned={FILTER_MARKER: '{"category": "cs.NONEXISTENT"}'}, default_response="final answer"
    )
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm)
    # Must not raise -- degrades to empty retrieved_chunk_ids / empty context.
    result = fn("Among the cs.NONEXISTENT papers, what speeds up inference?", k=5)
    assert result.retrieved_chunk_ids == []
    assert result.answer == "final answer"


def test_token_counts_summed_across_both_calls():
    llm = MockLLM(canned={FILTER_MARKER: '{"category": "cs.LG"}'}, default_response="final answer")
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm)
    result = fn("Among the cs.LG papers, what speeds up inference?", k=5)

    assert len(llm.calls) == 2
    expected_input = sum(max(1, len(c.prompt) // 4) for c in llm.calls)
    assert result.input_tokens == expected_input
