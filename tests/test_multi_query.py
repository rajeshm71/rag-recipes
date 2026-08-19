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


# Direct test of the best-rank merge logic in recipes/multi_query.py's
# `best_rank` dict -- a chunk ranked #1 by one sub-query and #5 by another
# must end up using rank #1 -- which the embed()-call-counting test above
# doesn't exercise. MockEmbedder's hash-based vectors
# aren't semantically controllable, so dense_search is stubbed directly
# with per-subquery results instead, isolating the merge arithmetic itself
# from embedding realism.
def test_merge_uses_best_rank_across_subqueries(monkeypatch):
    llm = MockLLM(
        canned={DECOMPOSE_MARKER: '{"queries": ["sub one", "sub two"]}'},
        default_response="final answer",
    )

    # sub one ranks: c1=1, c2=2, c3=3
    # sub two ranks: c3=1, c1=2, c2=3
    # best (lowest) rank per chunk: c1=1, c3=1, c2=2
    fake_results = {
        "sub one": ["c1", "c2", "c3"],
        "sub two": ["c3", "c1", "c2"],
    }

    def fake_dense_search(store, embedder, embedding_model, query_text, k):
        return fake_results[query_text]

    monkeypatch.setattr("recipes.multi_query.dense_search", fake_dense_search)

    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm)
    result = fn("some question", k=3)

    # c3 was only rank 3 under "sub one" but rank 1 under "sub two" -- its
    # best rank (1) must win, placing it ahead of c2 (best rank 2). Ties
    # (c1 and c3, both best rank 1) preserve first-seen order, since
    # `sorted()` is stable and c1 was inserted into best_rank first.
    assert result.retrieved_chunk_ids == ["c1", "c3", "c2"]


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
