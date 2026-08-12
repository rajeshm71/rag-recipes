"""Pattern 03: hybrid retrieval. Runs dense and BM25 independently, fuses
results via Reciprocal Rank Fusion (RRF). Beats either ranker alone almost
always, at roughly 2x the retrieval cost of a single ranker.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from recipes import AnswerWithCitations
from recipes.bm25 import BM25Index
from recipes.dense_index import build_dense_index, dense_search
from recipes.embeddings import Embedder
from recipes.llm import LLM

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4.1-mini-2025-04-14"
_GENERATION_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generation_prompt.txt"

RRF_K = 60  # standard literature default
CANDIDATES_PER_RANKER = 20


def rrf_fuse(ranked_lists: list[list[str]], k: int, rrf_k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion over any number of ranked chunk_id lists.

    Extracted from make_retrieve_and_answer's inline fusion loop (P5) so
    recipes/hybrid_rerank.py (used by the A1/A2 appendix studies) can reuse
    the exact same fusion math instead of duplicating it.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    fused = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [cid for cid, _ in fused[:k]]


def make_retrieve_and_answer(
    corpus_by_id: dict[str, dict],
    embedder: Embedder,
    llm: LLM,
    embedding_model: str = EMBEDDING_MODEL,
    generation_model: str = GENERATION_MODEL,
    rrf_k: int = RRF_K,
    candidates_per_ranker: int = CANDIDATES_PER_RANKER,
) -> Callable[..., AnswerWithCitations]:
    dense_store = build_dense_index(corpus_by_id, embedder, embedding_model)

    chunk_ids = list(corpus_by_id.keys())
    texts = [corpus_by_id[cid]["text"] for cid in chunk_ids]
    bm25_index = BM25Index()
    bm25_index.build(chunk_ids, texts)

    prompt_template = _GENERATION_PROMPT_PATH.read_text(encoding="utf-8")

    def retrieve_and_answer(question: str, k: int = 5) -> AnswerWithCitations:
        start = time.perf_counter()

        dense_ids = dense_search(
            dense_store, embedder, embedding_model, question, candidates_per_ranker
        )
        bm25_ids = [r.chunk_id for r in bm25_index.search(question, candidates_per_ranker)]

        retrieved_ids = rrf_fuse([dense_ids, bm25_ids], k, rrf_k)

        context = "\n\n".join(
            f"[{cid}] {corpus_by_id[cid]['text']}"
            for cid in retrieved_ids
            if cid in corpus_by_id
        )
        prompt = prompt_template.format(context=context, question=question)

        response = llm.complete(prompt=prompt, model=generation_model, temperature=0.0)
        latency_ms = (time.perf_counter() - start) * 1000

        return AnswerWithCitations(
            answer=response.text,
            retrieved_chunk_ids=retrieved_ids,
            latency_ms=latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_input_tokens=response.cached_input_tokens,
            cache_creation_input_tokens=response.cache_creation_input_tokens,
        )

    return retrieve_and_answer
