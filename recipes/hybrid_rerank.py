"""Shared "hybrid retrieval + cross-encoder rerank" retrieval-only helper.
Used by A1 (chunking study) and A2 (embedding swap), which both hold this
exact retrieval pattern constant per SPEC.md §7 and only vary chunking or
embedding model respectively. Retrieval-only (no generation, no LLM calls
at all) -- neither appendix needs an answer, just retrieved chunk_ids to
score against the eval set via paper_hit_at_k.
"""

from __future__ import annotations

from typing import Callable

from recipes.bm25 import BM25Index
from recipes.dense_index import build_dense_index, dense_search
from recipes.embeddings import Embedder
from recipes.hybrid import rrf_fuse
from recipes.rerank import Reranker, get_reranker

EMBEDDING_MODEL = "text-embedding-3-small"
CANDIDATES_PER_RANKER = 20
FUSED_CANDIDATES_K = 20  # how many RRF survivors get reranked


def build_retriever(
    corpus_by_id: dict[str, dict],
    embedder: Embedder,
    embedding_model: str = EMBEDDING_MODEL,
    reranker: Reranker | None = None,
    candidates_per_ranker: int = CANDIDATES_PER_RANKER,
    fused_candidates_k: int = FUSED_CANDIDATES_K,
) -> Callable[[str, int], list[str]]:
    dense_store = build_dense_index(corpus_by_id, embedder, embedding_model)
    chunk_ids = list(corpus_by_id.keys())
    bm25_index = BM25Index()
    bm25_index.build(chunk_ids, [corpus_by_id[cid]["text"] for cid in chunk_ids])
    reranker = reranker or get_reranker()

    def retrieve(question: str, k: int = 10) -> list[str]:
        dense_ids = dense_search(dense_store, embedder, embedding_model, question, candidates_per_ranker)
        bm25_ids = [r.chunk_id for r in bm25_index.search(question, candidates_per_ranker)]
        fused = rrf_fuse([dense_ids, bm25_ids], k=fused_candidates_k)
        candidate_pairs = [(cid, corpus_by_id[cid]["text"]) for cid in fused if cid in corpus_by_id]
        reranked = reranker.rerank(question, candidate_pairs, top_k=k)
        return [r.chunk_id for r in reranked]

    return retrieve
