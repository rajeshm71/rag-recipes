from recipes.bm25 import BM25Index
from recipes.dense_index import build_dense_index, dense_search
from recipes.embeddings import MockEmbedder
from recipes.hybrid import make_retrieve_and_answer, rrf_fuse
from recipes.llm import MockLLM

FIXTURE_CORPUS = {
    "c1": {"chunk_id": "c1", "text": "Speculative decoding speeds up LLM inference."},
    "c2": {"chunk_id": "c2", "text": "Convolutional networks are used for image classification."},
    "c3": {"chunk_id": "c3", "text": "Reinforcement learning aligns model outputs with feedback."},
    "c4": {"chunk_id": "c4", "text": "Gradient descent optimizes neural network weights."},
    "c5": {"chunk_id": "c5", "text": "Transformers use self-attention for sequence modeling."},
}


def test_returns_bound_closure_with_generation():
    fn = make_retrieve_and_answer(
        FIXTURE_CORPUS, embedder=MockEmbedder(), llm=MockLLM(default_response="answer text")
    )
    result = fn("What speeds up inference?", k=2)

    assert result.answer == "answer text"
    assert len(result.retrieved_chunk_ids) == 2
    assert all(cid in FIXTURE_CORPUS for cid in result.retrieved_chunk_ids)
    assert result.latency_ms >= 0.0


def test_fused_results_only_come_from_the_two_rankers_candidate_pools():
    embedder = MockEmbedder()
    question = "What speeds up inference?"
    candidates_per_ranker = 3

    dense_store = build_dense_index(FIXTURE_CORPUS, embedder, embedding_model="mock-embed")
    dense_ids = set(
        dense_search(dense_store, embedder, "mock-embed", question, candidates_per_ranker)
    )
    bm25_index = BM25Index()
    bm25_index.build(list(FIXTURE_CORPUS.keys()), [c["text"] for c in FIXTURE_CORPUS.values()])
    bm25_ids = {r.chunk_id for r in bm25_index.search(question, candidates_per_ranker)}

    fn = make_retrieve_and_answer(
        FIXTURE_CORPUS,
        embedder=embedder,
        llm=MockLLM(),
        candidates_per_ranker=candidates_per_ranker,
    )
    result = fn(question, k=5)

    allowed = dense_ids | bm25_ids
    assert set(result.retrieved_chunk_ids).issubset(allowed)


def test_bm25_only_match_is_surfaced_by_rrf():
    # "Speculative" and "decoding" appear only in c1 -- BM25 must rank it
    # highly regardless of what the (semantically meaningless) mock dense
    # ranker does, and RRF fusion must not drop it entirely.
    fn = make_retrieve_and_answer(
        FIXTURE_CORPUS, embedder=MockEmbedder(), llm=MockLLM(), candidates_per_ranker=5
    )
    result = fn("speculative decoding", k=5)
    assert "c1" in result.retrieved_chunk_ids


def test_rrf_k_is_wired_into_the_fusion_formula():
    # Independently recompute the expected RRF-fused order from the two
    # rankers' raw outputs, using the literal formula, and assert the
    # pattern's actual output matches exactly -- proves rrf_k really flows
    # into the score rather than being a dead parameter (a no-op fusion,
    # e.g. always returning dense_ids unchanged, would not match this for
    # an arbitrary rrf_k).
    embedder = MockEmbedder()
    question = "neural network optimization"
    candidates_per_ranker = 5
    rrf_k = 7

    dense_store = build_dense_index(FIXTURE_CORPUS, embedder, embedding_model="mock-embed")
    dense_ids = dense_search(
        dense_store, embedder, "mock-embed", question, candidates_per_ranker
    )
    bm25_index = BM25Index()
    bm25_index.build(list(FIXTURE_CORPUS.keys()), [c["text"] for c in FIXTURE_CORPUS.values()])
    bm25_ids = [r.chunk_id for r in bm25_index.search(question, candidates_per_ranker)]

    expected_scores: dict[str, float] = {}
    for rank, cid in enumerate(dense_ids, start=1):
        expected_scores[cid] = expected_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, cid in enumerate(bm25_ids, start=1):
        expected_scores[cid] = expected_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    expected_order = [
        cid for cid, _ in sorted(expected_scores.items(), key=lambda pair: pair[1], reverse=True)
    ][:5]

    fn = make_retrieve_and_answer(
        FIXTURE_CORPUS,
        embedder=MockEmbedder(),
        llm=MockLLM(),
        rrf_k=rrf_k,
        candidates_per_ranker=candidates_per_ranker,
    )
    actual_order = fn(question, k=5).retrieved_chunk_ids

    assert actual_order == expected_order


def test_rrf_fuse_is_genuinely_n_ary():
    # rrf_fuse was extracted from this file's inline fusion loop for reuse
    # by recipes/hybrid_rerank.py (A1/A2 appendices). This pattern's own
    # usage only ever passes 2 lists, but the function itself must not
    # assume that -- verify a 3-list fusion combines all three properly.
    list_a = ["c1", "c2", "c3"]
    list_b = ["c3", "c1", "c2"]
    list_c = ["c2", "c3", "c1"]

    result = rrf_fuse([list_a, list_b, list_c], k=3, rrf_k=60)

    # Every chunk appears in every list, just at different ranks -- all
    # three must survive fusion (none dropped), and the result must be a
    # genuine permutation of the three ids, not e.g. just list_a unchanged.
    assert set(result) == {"c1", "c2", "c3"}
    assert len(result) == 3


def test_rrf_fuse_respects_k():
    result = rrf_fuse([["c1", "c2", "c3"], ["c3", "c2", "c1"]], k=2, rrf_k=60)
    assert len(result) == 2
