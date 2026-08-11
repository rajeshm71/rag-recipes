from recipes.dense_index import build_dense_index, dense_search
from recipes.embeddings import MockEmbedder

FIXTURE_CORPUS = {
    "c1": {"chunk_id": "c1", "text": "Speculative decoding speeds up LLM inference."},
    "c2": {"chunk_id": "c2", "text": "Convolutional networks are used for image classification."},
    "c3": {"chunk_id": "c3", "text": "Reinforcement learning aligns model outputs with feedback."},
}


def test_build_dense_index_returns_searchable_store():
    embedder = MockEmbedder()
    store = build_dense_index(FIXTURE_CORPUS, embedder, embedding_model="mock-embed")
    retrieved = dense_search(store, embedder, "mock-embed", "What speeds up inference?", k=2)
    assert len(retrieved) == 2
    assert all(cid in FIXTURE_CORPUS for cid in retrieved)


def test_dense_search_returns_real_corpus_chunk_ids():
    embedder = MockEmbedder()
    store = build_dense_index(FIXTURE_CORPUS, embedder, embedding_model="mock-embed")
    retrieved = dense_search(store, embedder, "mock-embed", "any question", k=10)
    # k=10 exceeds corpus size (3); should return all 3, not error or pad.
    assert set(retrieved) == set(FIXTURE_CORPUS.keys())


def test_dimension_matches_non_default_embedder_dim():
    embedder = MockEmbedder(dim=8)
    store = build_dense_index(FIXTURE_CORPUS, embedder, embedding_model="mock-embed")
    assert store.dim == 8
    retrieved = dense_search(store, embedder, "mock-embed", "test question", k=1)
    assert len(retrieved) == 1
