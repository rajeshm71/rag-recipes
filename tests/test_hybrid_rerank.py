from recipes.bm25 import BM25Index
from recipes.dense_index import build_dense_index, dense_search
from recipes.embeddings import MockEmbedder
from recipes.hybrid_rerank import build_retriever
from recipes.rerank import MockReranker

FIXTURE_CORPUS = {
    "c1": {"chunk_id": "c1", "text": "Speculative decoding speeds up LLM inference."},
    "c2": {"chunk_id": "c2", "text": "Convolutional networks are used for image classification."},
    "c3": {"chunk_id": "c3", "text": "Reinforcement learning aligns model outputs with feedback."},
    "c4": {"chunk_id": "c4", "text": "Gradient descent optimizes neural network weights."},
    "c5": {"chunk_id": "c5", "text": "Transformers use self-attention for sequence modeling."},
}


def test_build_retriever_returns_bound_closure():
    retrieve = build_retriever(FIXTURE_CORPUS, embedder=MockEmbedder(), reranker=MockReranker())
    result = retrieve("What speeds up inference?", k=3)

    assert len(result) == 3
    assert all(cid in FIXTURE_CORPUS for cid in result)


def test_result_only_contains_candidates_from_the_two_rankers_pools():
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

    retrieve = build_retriever(
        FIXTURE_CORPUS,
        embedder=embedder,
        reranker=MockReranker(),
        candidates_per_ranker=candidates_per_ranker,
    )
    result = retrieve(question, k=5)

    allowed = dense_ids | bm25_ids
    assert set(result).issubset(allowed)


def test_works_end_to_end_with_mock_reranker_no_real_model_download():
    # MockReranker is injected explicitly -- proves this never needs to
    # construct a real CrossEncoderReranker in unit tests.
    retrieve = build_retriever(FIXTURE_CORPUS, embedder=MockEmbedder(), reranker=MockReranker())
    result = retrieve("speculative decoding", k=2)
    assert len(result) == 2
