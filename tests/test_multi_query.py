from recipes.embeddings import MockEmbedder
from recipes.llm import MockLLM
from recipes.multi_query import _parse_subqueries, make_retrieve_and_answer

FIXTURE_CORPUS = {
    "c1": {"chunk_id": "c1", "text": "Speculative decoding speeds up LLM inference."},
    "c2": {"chunk_id": "c2", "text": "Convolutional networks are used for image classification."},
    "c3": {"chunk_id": "c3", "text": "Reinforcement learning aligns model outputs with feedback."},
}

# Unique substring that only appears in the decomposition prompt template,
# never in the generation prompt template -- lets MockLLM's `canned` dict
# distinguish the two calls unambiguously.
DECOMPOSE_MARKER = "Respond with ONLY a JSON object"


def test_parse_subqueries_valid_json():
    assert _parse_subqueries('{"queries": ["a", "b"]}', fallback_question="q") == ["a", "b"]


def test_parse_subqueries_code_fenced_json():
    text = '```json\n{"queries": ["a", "b", "c"]}\n```'
    assert _parse_subqueries(text, fallback_question="q") == ["a", "b", "c"]


def test_parse_subqueries_garbage_text_falls_back():
    assert _parse_subqueries("not json at all", fallback_question="q") == ["q"]


def test_parse_subqueries_empty_list_falls_back():
    assert _parse_subqueries('{"queries": []}', fallback_question="q") == ["q"]


def test_parse_subqueries_missing_key_falls_back():
    assert _parse_subqueries('{"other": "field"}', fallback_question="q") == ["q"]


def test_llm_called_twice_and_dense_search_runs_per_subquery():
    llm = MockLLM(
        canned={DECOMPOSE_MARKER: '{"queries": ["query one", "query two"]}'},
        default_response="final answer",
    )
    embedder = MockEmbedder()
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=embedder, llm=llm)
    fn("some ambiguous question", k=2)

    # 2 LLM calls: decomposition + final generation.
    assert len(llm.calls) == 2

    # embed() calls: 1 for building the corpus index, then 1 per subquery
    # (2 subqueries here) for their individual searches.
    assert len(embedder.calls) == 1 + 2


def test_result_uses_final_call_answer():
    llm = MockLLM(
        canned={DECOMPOSE_MARKER: '{"queries": ["query one"]}'},
        default_response="final answer text",
    )
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm)
    result = fn("some question", k=2)
    assert result.answer == "final answer text"


def test_token_counts_are_summed_across_both_calls():
    llm = MockLLM(
        canned={DECOMPOSE_MARKER: '{"queries": ["query one", "query two"]}'},
        default_response="final answer text",
    )
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm)
    result = fn("some question", k=2)

    assert len(llm.calls) == 2
    expected_input = sum(max(1, len(c.prompt) // 4) for c in llm.calls)
    assert result.input_tokens == expected_input
