"""Shared dense-retrieval index building, used by every pattern that needs
vector search: 01 (naive dense), 03 (hybrid), 05 (HyDE), 06 (multi-query).
"""

from __future__ import annotations

from recipes.embeddings import Embedder
from recipes.store import VectorStore


def build_dense_index(
    corpus_by_id: dict[str, dict],
    embedder: Embedder,
    embedding_model: str,
) -> VectorStore:
    """Batch-embeds every chunk in corpus_by_id once and returns a populated
    VectorStore. Dimension is read from the real embedding output (differs
    between MockEmbedder's 16 dims and text-embedding-3-small's 1536),
    never hardcoded.
    """
    chunk_ids = list(corpus_by_id.keys())
    texts = [corpus_by_id[cid]["text"] for cid in chunk_ids]

    embed_response = embedder.embed(texts, model=embedding_model)
    dim = len(embed_response.vectors[0]) if embed_response.vectors else 0
    store = VectorStore(dim=dim)
    store.add_many(chunk_ids, embed_response.vectors)
    return store


def dense_search(
    store: VectorStore,
    embedder: Embedder,
    embedding_model: str,
    query_text: str,
    k: int,
) -> list[str]:
    """Embeds `query_text` (a raw question, OR for HyDE a hypothetical
    document, OR for multi-query a decomposed sub-query) and returns the
    top-k nearest chunk_ids.
    """
    query_embed = embedder.embed([query_text], model=embedding_model)
    results = store.search(query_embed.vectors[0], k=k)
    return [r.chunk_id for r in results]
