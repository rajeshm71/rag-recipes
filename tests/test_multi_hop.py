from recipes.embeddings import MockEmbedder
from recipes.llm import MockLLM
from recipes.multi_hop import _parse_next_query, make_retrieve_and_answer

FIXTURE_CORPUS = {
    "c1": {"chunk_id": "c1", "text": "Speculative decoding speeds up LLM inference."},
    "c2": {"chunk_id": "c2", "text": "Convolutional networks are used for image classification."},
    "c3": {"chunk_id": "c3", "text": "Reinforcement learning aligns model outputs with feedback."},
}

# Unique substring that only appears in multi_hop_prompt.txt.
HOP_MARKER = "answering a multi-hop question"


def test_parse_next_query_valid():
    assert _parse_next_query('{"next_query": "a follow-up query"}') == "a follow-up query"


def test_parse_next_query_null():
    assert _parse_next_query('{"next_query": null}') is None


def test_parse_next_query_garbage_falls_back_to_none():
    assert _parse_next_query("not json") is None


def test_parse_next_query_missing_key_falls_back_to_none():
    assert _parse_next_query('{"other": "field"}') is None


def test_early_termination_when_hop_says_null():
    llm = MockLLM(canned={HOP_MARKER: '{"next_query": null}'}, default_response="final answer")
    embedder = MockEmbedder()
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=embedder, llm=llm, max_hops=2)
    fn("some question", k=2)

    # embed() calls: 1 to build the index, 1 for hop 0's search. Hop 1
    # never runs because the LLM said next_query: null after hop 0.
    assert len(embedder.calls) == 1 + 1
    # LLM calls: 1 hop-decision call + 1 final generation call.
    assert len(llm.calls) == 2


def test_runs_all_hops_when_llm_keeps_requesting_more():
    llm = MockLLM(canned={HOP_MARKER: '{"next_query": "a follow-up query"}'}, default_response="final answer")
    embedder = MockEmbedder()
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=embedder, llm=llm, max_hops=3)
    fn("some question", k=2)

    # 3 hops of search (index build + 3 searches), 2 hop-decision calls
    # (hops 0 and 1; hop 2 is the last hop, no decision needed) + 1 final
    # generation call = 3 LLM calls total.
    assert len(embedder.calls) == 1 + 3
    assert len(llm.calls) == 3


def test_hop_2_uses_llm_provided_follow_up_query(monkeypatch):
    llm = MockLLM(canned={HOP_MARKER: '{"next_query": "follow up query text"}'}, default_response="final answer")

    queries_searched = []

    def fake_dense_search(store, embedder, embedding_model, query_text, k):
        queries_searched.append(query_text)
        return ["c1"]

    monkeypatch.setattr("recipes.multi_hop.dense_search", fake_dense_search)

    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm, max_hops=2)
    fn("the original question", k=2)

    assert queries_searched[0] == "the original question"
    assert queries_searched[1] == "follow up query text"


def test_token_counts_summed_across_variable_number_of_calls():
    llm = MockLLM(canned={HOP_MARKER: '{"next_query": "another query"}'}, default_response="final answer")
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm, max_hops=3)
    result = fn("some question", k=2)

    assert len(llm.calls) == 3
    expected_input = sum(max(1, len(c.prompt) // 4) for c in llm.calls)
    assert result.input_tokens == expected_input
